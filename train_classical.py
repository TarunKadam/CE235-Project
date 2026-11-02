import sys
import os
import time
import numpy as np
import joblib
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from common import (compute_features, ClusterClassifier, predict_map,
                    colorize, save_result, auto_step)
from data_prep import load_data

args = sys.argv[1:]
if len(args) not in (2, 4):
    sys.exit(__doc__)

rng = np.random.default_rng(42)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/maps", exist_ok=True)

rgb, lab, spl, test = load_data(*args)
F = compute_features(rgb)

if test is None:                                 
    Xtr, ytr = F[(spl == 1) | (spl == 2)], lab[(spl == 1) | (spl == 2)]  
    Xte, yte = F[spl == 3], lab[spl == 3]
    images = {"image": rgb}
else:                                           
    rgb_te, lab_te = test
    Xtr, ytr = F[spl > 0], lab[spl > 0]        
    m = lab_te > 0
    Xte, yte = compute_features(rgb_te)[m], lab_te[m]
    images = {"trainimg": rgb, "testimg": rgb_te}


def balanced_sample(X, y, cap):
    idx = []
    for c in np.unique(y):
        i = np.where(y == c)[0]
        idx.append(rng.choice(i, min(cap, len(i)), replace=False))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


if len(Xte) > 100_000:                       
    j = rng.choice(len(Xte), 100_000, replace=False)
    Xte, yte = Xte[j], yte[j]

models = {
    "logreg": (make_pipeline(StandardScaler(),
               LogisticRegression(max_iter=1000, class_weight="balanced")), 30_000),
    "svm": (make_pipeline(StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")), 3_000),
    "random_forest": (RandomForestClassifier(n_estimators=100, n_jobs=-1,
                      class_weight="balanced", random_state=42), 10_000),
    "kmeans": (ClusterClassifier(k=12), 15_000),
}

for name, (model, cap) in models.items():
    print(f"\n--- Training {name} ---")
    Xs, ys = balanced_sample(Xtr, ytr, cap)     
    t0 = time.time()
    model.fit(Xs, ys)
    train_time = time.time() - t0
    save_result(name, yte, model.predict(Xte), train_time)
    joblib.dump(model, f"models/{name}.joblib")

    print(f"Generating prediction maps for {name}...")
    for key, img in images.items():
        h, w, _ = img.shape
        step = auto_step(h, w, limit=300_000)
        pred_map = predict_map(model, img, step=step)
        Image.fromarray(colorize(pred_map)).save(
            f"outputs/maps/{name}_map_{key}.png")

print("\nDone. Models in ./models, metrics + maps in ./outputs")