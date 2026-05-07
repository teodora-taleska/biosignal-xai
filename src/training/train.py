import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import CFG

def train_model(model, train_ds, val_ds, epochs=CFG['training']['epochs'], lr=1e-3, batch_size=CFG['training']['batch_size'], save_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    if save_dir is None:
        save_dir = os.path.join(CFG['paths']['results'], 'baseline_cnn')
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, 'checkpoint.pt')

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # BCEWithLogitsLoss for multi-label
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 4

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        scheduler.step()

        tl = train_loss / len(train_loader)
        vl = val_loss   / len(val_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f}")

        # Save best model / early stopping
        if vl < best_val_loss:
            best_val_loss = vl
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  * Saved -> {ckpt_path}  (val_loss={vl:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered (no improvement for {patience} epochs)")
                break

    return model