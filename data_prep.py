import numpy as np
import pandas as pd

CLASS_COLUMN = None        
CANDIDATES = ["class_id", "classid", "class", "label", "landcover", "lulc",
              "type", "category", "name", "id"]
NAME_CANDIDATES = ("class_name", "classname", "label_name", "name")
VAL_FRACTION = 0.10    


def _read_raster(path):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]) if src.count >= 3 else np.repeat(src.read(1)[None], 3, 0)
        transform, crs, bounds = src.transform, src.crs, src.bounds
    if arr.dtype != np.uint8:
        hi = max(float(np.percentile(arr, 99)), 1.0)
        arr = (np.clip(arr / hi, 0, 1) * 255).astype(np.uint8)
    rgb = np.transpose(arr, (1, 2, 0)).copy()
    print(f"  image {path}: {rgb.shape[1]}x{rgb.shape[0]} px | CRS: {crs}")
    return rgb, transform, crs, bounds


def _read_polys(path, crs, bounds, tag):
    import geopandas as gpd
    from shapely.geometry import box
    gdf = gpd.read_file(path)
    cols = [c for c in gdf.columns if c != "geometry"]
    print(f"  [{tag}] {path}: {len(gdf)} polygons | columns: {cols} | CRS: {gdf.crs}")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)

    if crs is None:
        print(f"  WARNING [{tag}]: image has no CRS; assuming polygons use the image's coordinates.")
    elif gdf.crs is None:
        print(f"  WARNING [{tag}]: polygons have no CRS; assuming the image's CRS.")
        gdf = gdf.set_crs(crs)
    else:
        gdf = gdf.to_crs(crs)

    if not gdf.intersects(box(*bounds)).any():
        raise SystemExit(
            f"\nERROR [{tag}]: none of your polygons overlap the image.\n"
            f"  image bounds  : {tuple(round(b, 5) for b in bounds)}\n"
            f"  polygon bounds: {tuple(round(b, 5) for b in gdf.total_bounds)}\n"
            "Re-export this image and its polygons from QGIS with the SAME CRS. "
            "(Each image must be paired with the polygons drawn on it.)")

    col = CLASS_COLUMN or next((c for cand in CANDIDATES for c in cols
                                if c.lower() == cand), None)
    if col is None and len(cols) == 1:
        col = cols[0]
        print(f"  [{tag}] using the only attribute column '{col}' as the class column.")
    if col is None:
        raise SystemExit(f"\nERROR [{tag}]: could not find a class column in {cols}.\n"
                         "Set CLASS_COLUMN at the top of data_prep.py to the right one.")
    name_col = next((c for c in cols if c != col and c.lower() in NAME_CANDIDATES), None)
    gdf = gdf.dropna(subset=[col]).reset_index(drop=True)
    return gdf, col, name_col


def _build_classes(parts):
    import common
    uniq = []
    for gdf, col, _ in parts:
        uniq += list(pd.unique(gdf[col]))
    uniq = list(dict.fromkeys(uniq))
    try:
        uniq = sorted(uniq)
    except TypeError:
        uniq = sorted(uniq, key=str)
    mapping = {v: i + 1 for i, v in enumerate(uniq)}  

    lookup = {}
    for gdf, col, name_col in parts:
        if name_col:
            for k, v in gdf.groupby(col)[name_col].first().to_dict().items():
                if pd.notna(v):
                    lookup.setdefault(k, str(v))
    names = {}
    for v, i in mapping.items():
        if v in lookup:
            names[i] = lookup[v]
        elif isinstance(v, (int, np.integer, float, np.floating)):
            names[i] = common.CLASS_NAMES.get(int(v), f"Class {v}")
        else:
            names[i] = str(v)
    common.set_classes(names)
    for gdf, col, _ in parts:
        gdf["class_id"] = gdf[col].map(mapping).astype(int)
        gdf["poly_id"] = np.arange(len(gdf))
    print(f"  Classes: {names}")
    return names


def _rasterize(gdf, values, shape, transform):
    from rasterio.features import rasterize
    return rasterize([(g, int(v)) for g, v in zip(gdf.geometry, values)],
                     out_shape=shape, transform=transform, fill=0, dtype="uint8")


def _split(gdf, lab, transform, seed, val_frac, test_frac, min_poly):
    from sklearn.model_selection import train_test_split
    H, W = lab.shape
    n_per = gdf["class_id"].value_counts()
    if n_per.min() >= min_poly:                       
        meta = gdf[["poly_id", "class_id"]]
        rest = val_frac + test_frac
        train, temp = train_test_split(meta, test_size=rest,
                                       stratify=meta["class_id"], random_state=seed)
        code = {pid: 1 for pid in train["poly_id"]}
        if test_frac > 0:
            val, test = train_test_split(temp, test_size=test_frac / rest,
                                         stratify=temp["class_id"], random_state=seed)
            code.update({pid: 2 for pid in val["poly_id"]})
            code.update({pid: 3 for pid in test["poly_id"]})
        else:
            code.update({pid: 2 for pid in temp["poly_id"]})
        spl = _rasterize(gdf, [code[p] for p in gdf["poly_id"]], (H, W), transform)
        print("  Split: by polygon.")
    else:                                               
        B = 64
        nbx, nby = -(-W // B), -(-H // B)
        blk = ((np.arange(H) // B).astype(np.int32)[:, None] * nbx
               + (np.arange(W) // B).astype(np.int32)[None, :])
        used = np.unique(blk[lab > 0])
        np.random.default_rng(seed).shuffle(used)
        a = int((1 - val_frac - test_frac) * len(used))
        b = int((1 - test_frac) * len(used))
        lut = np.zeros(nby * nbx, dtype=np.uint8)
        lut[used[:a]], lut[used[a:b]], lut[used[b:]] = 1, 2, 3
        spl = np.where(lab > 0, lut[blk], 0).astype(np.uint8)
        print(f"  Split: by {B}x{B}-pixel spatial blocks "
              f"(fewer than {min_poly} polygons in some class).")
    return spl


def _report(title, lab, mask_or_code, names):
    n = len(names)
    counts = np.bincount(lab[mask_or_code], minlength=n + 1)[1:]
    print(f"  {title:11s}", {names[i + 1]: int(c) for i, c in enumerate(counts)})
    return counts


def load_data(img_path, poly_path, test_img_path=None, test_poly_path=None, seed=42):
    two = test_img_path is not None
    if two and test_poly_path is None:
        raise SystemExit("ERROR: a test image was given without its polygon file.")

    print("Loading training image:")
    rgb, transform, crs, bounds = _read_raster(img_path)
    gdf, col, ncol = _read_polys(poly_path, crs, bounds, "train")
    parts = [(gdf, col, ncol)]
    if two:
        print("Loading test image:")
        rgb_t, tr_t, crs_t, b_t = _read_raster(test_img_path)
        gdf_t, col_t, ncol_t = _read_polys(test_poly_path, crs_t, b_t, "test")
        parts.append((gdf_t, col_t, ncol_t))

    names = _build_classes(parts)
    print("  Training polygons per class:",
          {names[k]: int(v) for k, v in gdf["class_id"].value_counts().sort_index().items()})

    H, W, _ = rgb.shape
    lab = _rasterize(gdf, gdf["class_id"], (H, W), transform)
    if (lab > 0).sum() == 0:
        raise SystemExit("\nERROR: training polygons cover no image pixels. "
                         "Check that image and polygons share the same CRS.")

    if two:
        spl = _split(gdf, lab, transform, seed, val_frac=VAL_FRACTION, test_frac=0.0, min_poly=10)
        lab_t = _rasterize(gdf_t, gdf_t["class_id"], rgb_t.shape[:2], tr_t)
        if (lab_t > 0).sum() == 0:
            raise SystemExit("\nERROR: test polygons cover no test-image pixels.")
        test = (rgb_t, lab_t)
    else:
        spl = _split(gdf, lab, transform, seed, val_frac=0.15, test_frac=0.15, min_poly=7)
        test = None

    print("Pixel counts:")
    c_tr = _report("train", lab, spl == 1, names)
    c_va = _report("val", lab, spl == 2, names)
    if two:
        c_te = _report("test image", test[1], test[1] > 0, names)
    else:
        c_te = _report("test", lab, spl == 3, names)

    if c_tr.sum() == 0 or c_va.sum() == 0 or c_te.sum() == 0:
        raise SystemExit("\nERROR: the train, validation or test set is empty. Draw more polygons "
                         "(spread over different parts of the image).")
    for i, (a, b) in enumerate(zip(c_tr, c_te)):
        if a == 0:
            print(f"  WARNING: class '{names[i + 1]}' has no training pixels.")
        if b == 0:
            print(f"  WARNING: class '{names[i + 1]}' has no test pixels "
                  "(its test metrics will be 0 / undefined).")
    return rgb, lab, spl, test