# -*- coding: utf-8 -*-
"""TransUnet.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/161e2-EUPhvEzs_b6GWyOWPfUNmAeWSas
"""

# Cell 1: Force fork start method (must be first!)
#import multiprocessing as mp
#mp.set_start_method('fork', force=True)

# Cell 2: Install dependencies
!pip install segmentation-models-pytorch timm imagecodecs tifffile

# Cell 2: Install dependencies (with TIFF codecs)
#!pip install segmentation-models-pytorch timm tifffile[all] imagecodecs

# Cell 3: Imports & Drive mount
import imagecodecs  # enable LZW TIFF support
import os
import pandas as pd
from google.colab import drive
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tifffile import imread
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F

drive.mount('/content/drive', force_remount=True)

# Cell 1 ▶ Dataset (224×224 targets, no down-scaling)
class LSTDataset(Dataset):
    def __init__(self, df, patches_dir, weather_cols):
        self.df           = df.reset_index(drop=True)
        self.patches_dir  = patches_dir
        self.weather_cols = weather_cols
        self.transform    = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),          # resize image
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.loc[idx]
        arr  = imread(os.path.join(self.patches_dir, row["filename"])
                     ).astype(np.float32)            # (4,H,W)

        # --- inputs -------------------------------------------------
        img_np = arr[[1,2,3]].transpose(1,2,0).astype(np.uint8)
        img    = self.transform(img_np)              # [3,224,224]

        # --- target (LST) at *full* 224×224 --------------------------
        lst    = arr[0]                              # (H,W)
        lst    = torch.tensor(lst, dtype=torch.float32).unsqueeze(0)
        lst    = F.interpolate(lst.unsqueeze(0), size=(224,224),
                               mode='bilinear', align_corners=False
                              ).squeeze(0)           # [1,224,224]

        # --- meteorology vector -------------------------------------
        weather = torch.tensor(
            row[self.weather_cols].values.astype(np.float32)
        )
        return img, weather, lst

# New Cell (before Cell 3)
!pip install rasterio

# define which meteorological columns to pull
weather_cols = [
    "air_temp_C",
    "dew_point_C",
    "relative_humidity_percent",
    "wind_speed_m_s",
    "precipitation_in",
]

import pandas as pd
import numpy as np

df = pd.read_csv("/content/drive/MyDrive/PatchedOutput/tiff_with_meteo.csv")
for col in weather_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=weather_cols + ["filename"]).reset_index(drop=True)

patches_dir = "/content/drive/MyDrive/PatchedOutput_Cleaned"
dataset     = LSTDataset(df, patches_dir, weather_cols)
train_sz    = int(0.8 * len(dataset))
val_sz      = len(dataset) - train_sz
train_ds, val_ds = random_split(dataset, [train_sz, val_sz])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0, pin_memory=False)

# Cell 5: TransUNet-like model (MiT-B0 backbone) + Focal-Tversky + optimizer
import segmentation_models_pytorch as smp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# UNet with MiT-B0 backbone (transformer-based encoder)
model = smp.Unet(
    encoder_name='mit_b0',         # <-- use a supported MiT backbone
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
).to(DEVICE)

def focal_tversky_loss(logits, targets,
                       alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims  = (2,3)
    TP = (probs * targets).sum(dim=dims)
    FN = ((1-probs) * targets).sum(dim=dims)
    FP = (probs * (1-targets)).sum(dim=dims)

    denom = TP + alpha*FN + beta*FP + eps
    denom = torch.clamp(denom, min=eps)
    tversky = (TP + eps) / denom
    return ((1 - tversky) ** gamma).mean()

optimizer = optim.Adam(model.parameters(), lr=1e-5)
scaler    = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# Cell 6: Training loop w/ live tqdm, grad-clip, no NaNs
NUM_EPOCHS = 10
best_val   = float('inf')

for epoch in range(1, NUM_EPOCHS+1):
    print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")

    # — Training —
    train_losses = []
    model.train()
    train_bar = tqdm(train_loader, desc='Train', leave=False)
    for imgs, _, masks in train_bar:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        masks = torch.clamp(masks, 0.0, 1.0)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = model(imgs)
            if masks.shape[2:] != preds.shape[2:]:
                masks = F.interpolate(
                    masks, size=preds.shape[2:], mode='nearest'
                )
            loss = focal_tversky_loss(preds, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train = sum(train_losses) / len(train_losses)

    # — Validation —
    val_losses = []
    model.eval()
    valid_bar = tqdm(val_loader, desc='Valid', leave=False)
    with torch.no_grad():
        for imgs, _, masks in valid_bar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            masks = torch.clamp(masks, 0.0, 1.0)

            preds = model(imgs)
            if masks.shape[2:] != preds.shape[2:]:
                masks = F.interpolate(
                    masks, size=preds.shape[2:], mode='nearest'
                )
            loss = focal_tversky_loss(preds, masks)

            val_losses.append(loss.item())
            valid_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_val = sum(val_losses) / len(val_losses)
    print(f"Epoch {epoch}/{NUM_EPOCHS} → "
          f"Avg Train: {avg_train:.4f} | Avg Val: {avg_val:.4f}")

    scheduler.step(avg_val)
    if avg_val < best_val:
        best_val = avg_val
        torch.save(
            model.state_dict(),
            "/content/drive/MyDrive/best_transunet_ftl.pth"
        )
        print("🔖 Saved new best model!")

print("✅ Training complete.")