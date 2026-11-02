import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from common import CLASS_IDS, colorize, save_result
from data_prep import load_data
from cnn_model import SmallCNN, to_tensor, predict_map_cnn, PATCH

args = sys.argv[1:]
if len(args) not in (2, 4):
    sys.exit(__doc__)

EPOCHS, BATCH, LR = 25, 128, 1e-3
rng = np.random.default_rng(42)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/maps", exist_ok=True)

rgb, lab, spl, test = load_data(*args)
half = PATCH // 2


def pad(img):
    return np.pad(img, ((half, half), (half, half), (0, 0)), mode="reflect")


def sample(padded, labels, mask, per_class):
    X, y = [], []
    for c in CLASS_IDS:
        rr, cc = np.where(mask & (labels == c))
        if len(rr) == 0:
            continue
        idx = rng.choice(len(rr), min(per_class, len(rr)), replace=False)
        for r, k in zip(rr[idx], cc[idx]):
            X.append(padded[r:r + PATCH, k:k + PATCH])
            y.append(c - 1)
    return np.stack(X), np.array(y)


padded = pad(rgb)
Xtr, ytr = sample(padded, lab, spl == 1, 4000)
Xva, yva = sample(padded, lab, spl == 2, 1500)
if test is None:
    Xte, yte = sample(padded, lab, spl == 3, 3000)
    images = {"image": rgb}
else:
    rgb_te, lab_te = test
    Xte, yte = sample(pad(rgb_te), lab_te, lab_te > 0, 3000)
    images = {"trainimg": rgb, "testimg": rgb_te}
print("patches (train / val / test):", Xtr.shape, Xva.shape, Xte.shape)

Ttr, Tva, Tte = to_tensor(Xtr), to_tensor(Xva), to_tensor(Xte)
Ytr = torch.from_numpy(ytr).long()

counts = np.bincount(ytr, minlength=len(CLASS_IDS)).astype(np.float32)
weights = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1))).to(device)

model = SmallCNN(len(CLASS_IDS)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(weight=weights)


def augment(x):
    if torch.rand(1) < 0.5:
        x = torch.flip(x, [3])
    if torch.rand(1) < 0.5:
        x = torch.flip(x, [2])
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), [2, 3])


def predict(T):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(T), 512):
            out.append(model(T[i:i + 512].to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()


best, t0 = 0.0, time.time()
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(Ttr))
    total = 0.0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i + BATCH]
        xb, yb = augment(Ttr[b]).to(device), Ytr[b].to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
    val_acc = (predict(Tva) == yva).mean()
    print(f"epoch {ep + 1:2d}  loss {total / len(perm):.4f}  val_acc {val_acc:.4f}")
    if val_acc > best:
        best = val_acc
        torch.save(model.state_dict(), "models/cnn.pt")
train_time = time.time() - t0

model.load_state_dict(torch.load("models/cnn.pt", map_location=device))
save_result("cnn", yte + 1, predict(Tte) + 1, train_time)

for key, img in images.items():
    Image.fromarray(colorize(predict_map_cnn(model, img, stride=8, device=device))).save(
        f"outputs/maps/cnn_map_{key}.png")
print("Done. Saved models/cnn.pt and maps in outputs/maps")