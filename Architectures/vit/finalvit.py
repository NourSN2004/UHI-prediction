# -*- coding: utf-8 -*-
"""FINALVIT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ht-CYyX1jwkL6UaMAZNqiDrBIPZx8M21
"""

from google.colab import drive
drive.mount('/content/drive')

# Cell 1 ▶ Install required packages
!pip install timm torchvision tifffile imagecodecs --quiet

# Cell 2 ▶ Consolidated imports
import os
import math
import time

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import timm
from torchvision import transforms, models

from tifffile import imread

class LSTDataset(Dataset):
    def __init__(self, df, patches_dir, weather_cols):
        self.df           = df.reset_index(drop=True)
        self.patches_dir  = patches_dir
        self.weather_cols = weather_cols
        self.transform    = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std =[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.loc[idx]
        path  = os.path.join(self.patches_dir, row["filename"])
        arr   = imread(path).astype(np.float32)  # shape = (bands, H, W)

        img_np = arr[[1,2,3],:,:].transpose(1,2,0).astype(np.uint8)
        img    = self.transform(img_np)  # [3,224,224]

        tar_np = arr[0,:,:]
        target = torch.tensor(tar_np, dtype=torch.float32).unsqueeze(0)  # [1,H,W]

        # ✅ Resize your target to match model output shape (56x56 exactly)
        target = F.interpolate(target.unsqueeze(0), size=(56, 56),
                               mode='bilinear', align_corners=False).squeeze(0)

        weather = torch.tensor(
            row[self.weather_cols].values.astype(np.float32)
        )

        return img, weather, target

# define which meteorological columns to pull
weather_cols = [
    "air_temp_C",
    "dew_point_C",
    "relative_humidity_percent",
    "wind_speed_m_s",
    "precipitation_in",
]

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

# 1) Define your model class (you only need to run this once per restart)
import math, torch.nn as nn, timm

class PretrainedViTLSTModel(nn.Module):
    def __init__(self,
                 weather_dim: int = 5,
                 hidden_dim:  int = 768,
                 vit_name:    str = "vit_base_patch16_224",
                 num_layers:  int = 2,
                 num_heads:   int = 8):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=True, num_classes=0)
        for p in self.vit.parameters(): p.requires_grad = False
        self.weather_proj = nn.Linear(weather_dim, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                         nhead=num_heads,
                                         dim_feedforward=hidden_dim*4,
                                         dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim//2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim//4), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//4, 1, kernel_size=1)
        )

    def forward(self, images, weather):
        feats   = self.vit.forward_features(images)   # [B,1+N,D]
        cls_tok = feats[:,0:1]                        # [B,1,D]
        patches = feats[:,1:,:]                       # [B,N,D]
        w_emb   = self.weather_proj(weather).unsqueeze(1)  # [B,1,D]
        tokens  = torch.cat([patches, w_emb, cls_tok], dim=1)  # [B,N+2,D]
        t       = tokens.permute(1,0,2)
        t       = self.transformer(t)
        t       = t.permute(1,0,2)
        patch_out = t[:,:-2,:]                        # drop weather+CLS
        B,N,D     = patch_out.shape
        G         = int(math.sqrt(N))
        x         = patch_out.transpose(1,2).view(B,D,G,G)
        return self.deconv(x)                         # [B,1,224,224]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = PretrainedViTLSTModel(
    weather_dim=len(weather_cols),
    hidden_dim=768,
    vit_name="vit_base_patch16_224",
    num_layers=2,
    num_heads=8
).to(device)

# unfreeze last ViT blocks:
for name, param in model.vit.named_parameters():
    if any(layer in name for layer in ["blocks.10", "blocks.11", "norm"]):
        param.requires_grad = True

opt       = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4, weight_decay=1e-2
)
loss_fn   = nn.SmoothL1Loss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=3, verbose=True
)

from tqdm import tqdm
import os
from pathlib import Path
import torch

# Create a directory on your Drive for checkpoints
save_dir = Path("/content/drive/MyDrive/ModelCheckpoints")
save_dir.mkdir(parents=True, exist_ok=True)

num_epochs = 10
start_ep   = 0  # or pick up from a checkpoint

for epoch in range(start_ep, num_epochs):
    # — Train —
    model.train()
    train_loss   = 0.0
    seen_samples = 0
    train_bar    = tqdm(train_loader, desc=f"Epoch {epoch+1:02d} Train")
    for imgs, weather, tgt in train_bar:
        imgs, weather, tgt = imgs.to(device), weather.to(device), tgt.to(device)

        opt.zero_grad()
        out  = model(imgs, weather)
        loss = loss_fn(out, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        batch_val     = loss.item()
        n             = imgs.size(0)
        train_loss   += batch_val * n
        seen_samples += n
        avg_train     = train_loss / seen_samples

        train_bar.set_postfix(
            batch_loss=f"{batch_val:.4f}",
            avg_loss  =f"{avg_train:.4f}"
        )

    train_rmse = (train_loss / len(train_loader.dataset))**0.5

    # — Validate —
    model.eval()
    val_loss = 0.0
    seen_val = 0
    val_bar  = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}   Val ")
    with torch.no_grad():
        for imgs, weather, tgt in val_bar:
            imgs, weather, tgt = imgs.to(device), weather.to(device), tgt.to(device)
            out       = model(imgs, weather)
            batch_val = loss_fn(out, tgt).item()
            n         = imgs.size(0)
            val_loss += batch_val * n
            seen_val += n
            avg_val   = val_loss / seen_val

            val_bar.set_postfix(
                batch_loss=f"{batch_val:.4f}",
                avg_loss  =f"{avg_val:.4f}"
            )

    val_rmse = (val_loss / len(val_loader.dataset))**0.5

    # Step the scheduler
    scheduler.step(val_loss)

    # Print metrics
    print(f"Epoch {epoch+1:02d} ▶ Train RMSE: {train_rmse:.3f} | Val RMSE: {val_rmse:.3f}")

    # — Save checkpoint —
    ckpt_path = save_dir / f"cnn_mlp_epoch{epoch+1:02d}.pt"
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_rmse': train_rmse,
        'val_rmse': val_rmse
    }, ckpt_path)
    print(f"✅ Saved checkpoint: {ckpt_path}")

print("✅ Training finished")

!find /content -type f -name "*checkpoint*" -print