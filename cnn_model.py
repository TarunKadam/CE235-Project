import numpy as np
import torch
import torch.nn as nn

PATCH = 32


class SmallCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))  # 32 -> 4
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def to_tensor(patches):
    t = torch.from_numpy(patches).permute(0, 3, 1, 2).float() / 255.0
    return (t - 0.5) / 0.5


def predict_map_cnn(model, rgb, stride=8, device="cpu", batch=512):
    half = PATCH // 2
    H, W, _ = rgb.shape
    padded = np.pad(rgb, ((half, half), (half, half), (0, 0)), mode="reflect")
    rows, cols = np.arange(0, H, stride), np.arange(0, W, stride)
    coords = [(r, c) for r in rows for c in cols]
    preds = []
    model.eval().to(device)
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            b = np.stack([padded[r:r + PATCH, c:c + PATCH] for r, c in coords[i:i + batch]])
            preds.append(model(to_tensor(b).to(device)).argmax(1).cpu().numpy())
    coarse = (np.concatenate(preds) + 1).astype(np.uint8).reshape(len(rows), len(cols))
    full = np.repeat(np.repeat(coarse, stride, axis=0), stride, axis=1)
    return full[:H, :W]
