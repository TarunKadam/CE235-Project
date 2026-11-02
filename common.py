import os
import json
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from skimage.color import rgb2hsv
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, classification_report)

CLASS_NAMES = {1: "vegetation", 2: "road", 3: "river", 4: "barren_land", 5: "Built-up"}
CLASS_COLORS = {1: (40, 160, 60), 2: (120, 120, 120), 3: (40, 100, 220),
                4: (210, 180, 110), 5: (220, 50, 50)}

CLASS_IDS = sorted(CLASS_NAMES)
CLASS_JSON = "outputs/classes.json"
_PALETTE = [(220, 50, 50), (120, 120, 120), (40, 160, 60), (40, 100, 220),
            (210, 180, 110), (150, 60, 180), (240, 150, 30), (30, 190, 190),
            (90, 50, 20), (250, 220, 40)]


def set_classes(names):
    CLASS_NAMES.clear()
    CLASS_NAMES.update(names)
    CLASS_IDS[:] = sorted(names)
    CLASS_COLORS.clear()
    CLASS_COLORS.update({k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(CLASS_IDS)})
    os.makedirs("outputs", exist_ok=True)
    with open(CLASS_JSON, "w") as f:
        json.dump({str(k): {"name": CLASS_NAMES[k], "color": list(CLASS_COLORS[k])}
                   for k in CLASS_IDS}, f)


if os.path.exists(CLASS_JSON):    
    with open(CLASS_JSON) as _f:
        _d = json.load(_f)
    CLASS_NAMES.clear()
    CLASS_NAMES.update({int(k): v["name"] for k, v in _d.items()})
    CLASS_COLORS.clear()
    CLASS_COLORS.update({int(k): tuple(v["color"]) for k, v in _d.items()})
    CLASS_IDS[:] = sorted(CLASS_NAMES)


def compute_features(rgb):
    x = rgb.astype(np.float32) / 255.0
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    eps = 1e-6
    gray = x.mean(axis=2)
    hsv = rgb2hsv(x)
    exg = 2 * G - R - B                     
    bmr = (B - R) / (B + R + eps)       
    feats = [R, G, B, gray, exg, bmr, hsv[..., 0], hsv[..., 1], hsv[..., 2]]
    for k in (3, 5):                   
        for c in (R, G, B):
            feats.append(uniform_filter(c, size=k))
    m = uniform_filter(gray, 5)        
    m2 = uniform_filter(gray ** 2, 5)
    feats.append(np.sqrt(np.clip(m2 - m * m, 0, None)))
    return np.stack(feats, axis=-1).astype(np.float32)


class ClusterClassifier:
    def __init__(self, k=12):
        self.k = k

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        self.km = MiniBatchKMeans(self.k, random_state=42, n_init=3,
                                  batch_size=4096).fit(self.scaler.transform(X))
        self.map = {}
        for i in range(self.k):
            m = self.km.labels_ == i
            self.map[i] = int(np.bincount(y[m]).argmax()) if m.any() else int(y[0])
        return self

    def predict(self, X):
        c = self.km.predict(self.scaler.transform(X))
        return np.array([self.map[i] for i in c], dtype=np.uint8)


def auto_step(h, w, limit=400_000):
    return max(1, int(np.ceil(np.sqrt(h * w / limit))))


def predict_map(model, rgb, chunk=200000, step=None):
    h, w, _ = rgb.shape
    if step is None:
        step = auto_step(h, w)

    rgb_s = rgb[::step, ::step] if step > 1 else rgb
    F = compute_features(rgb_s)
    hs, ws, f = F.shape
    X = F.reshape(-1, f)
    out = np.empty(len(X), dtype=np.uint8)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = model.predict(X[i:i + chunk])
    out = out.reshape(hs, ws)
    if step > 1:
        out = np.repeat(np.repeat(out, step, axis=0), step, axis=1)[:h, :w]
    return out


def colorize(label_map):
    lut = np.zeros((max(CLASS_IDS) + 1, 3), dtype=np.uint8)
    for k, v in CLASS_COLORS.items():
        lut[k] = v
    return lut[label_map]


def evaluate(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "kappa": cohen_kappa_score(y_true, y_pred)}


def save_result(name, y_true, y_pred, train_time, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    row = {"model": name, **evaluate(y_true, y_pred),
           "train_time_s": round(train_time, 1)}
    p = f"{out_dir}/results.csv"
    new = pd.DataFrame([row])
    if os.path.exists(p):
        old = pd.read_csv(p)
        old = old[old["model"] != name]
        new = pd.concat([old, new], ignore_index=True) if len(old) else new
    new.to_csv(p, index=False)

    jp = f"{out_dir}/confusion.json"
    d = json.load(open(jp)) if os.path.exists(jp) else {}
    d[name] = confusion_matrix(y_true, y_pred, labels=CLASS_IDS).tolist()
    json.dump(d, open(jp, "w"))

    print(f"\n=== {name} (TEST set) ===")
    print({k: round(v, 4) for k, v in row.items() if k != "model"})
    print(classification_report(y_true, y_pred, labels=CLASS_IDS,
                                target_names=[CLASS_NAMES[c] for c in CLASS_IDS],
                                zero_division=0))