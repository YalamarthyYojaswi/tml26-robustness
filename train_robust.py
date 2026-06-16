"""
Adversarial Training for Robustness — TML 2026
Method: PGD Adversarial Training (Madry et al., 2018)
Architecture: Standard ResNet18 (server-compatible)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18
from pathlib import Path

BASE       = Path("/home/atml_team024/tml26_task3")
DATA_PATH  = BASE / "train.npz"
MODEL_PATH = BASE / "model.pt"

NUM_CLASSES    = 9
BATCH_SIZE     = 128
EPOCHS         = 150
LR             = 0.05
WEIGHT_DECAY   = 5e-4
MOMENTUM       = 0.9
EPS            = 8  / 255.0
ALPHA          = 2  / 255.0
PGD_STEPS      = 10
PGD_STEPS_EVAL = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def build_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def pgd_attack(model, images, labels, eps, alpha, steps):
    delta = torch.zeros_like(images).uniform_(-eps, eps)
    delta = torch.clamp(delta, 0 - images, 1 - images)
    delta.requires_grad_(True)
    for _ in range(steps):
        loss = F.cross_entropy(model(images + delta), labels)
        loss.backward()
        with torch.no_grad():
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -eps, eps)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images
        delta.grad.zero_()
    return (images + delta).detach()

def augment(images):
    if torch.rand(1).item() > 0.5:
        images = torch.flip(images, dims=[3])
    padded = F.pad(images, (4, 4, 4, 4), mode='reflect')
    i = torch.randint(0, 8, (1,)).item()
    j = torch.randint(0, 8, (1,)).item()
    return padded[:, :, i:i+32, j:j+32]

def evaluate(model, loader, adversarial=False):
    model.eval()
    correct, total = 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        if adversarial:
            imgs = pgd_attack(model, imgs, lbls, EPS, ALPHA, PGD_STEPS_EVAL)
        with torch.no_grad():
            preds = model(imgs).argmax(1)
        correct += (preds == lbls).sum().item()
        total   += len(lbls)
    return correct / total

print("Loading dataset...")
data   = np.load(DATA_PATH)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()
print(f"  Images: {images.shape}, Labels: {labels.shape}")

dataset  = TensorDataset(images, labels)
n_val    = 5000
n_train  = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=256,
                          shuffle=False, num_workers=4, pin_memory=True)
print(f"  Train: {n_train}, Val: {n_val}")

model     = build_model().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                            nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[75, 110, 135], gamma=0.1)
criterion = nn.CrossEntropyLoss()

best_score = 0.0
print(f"\nStarting PGD adversarial training for {EPOCHS} epochs...")
print(f"  eps={EPS:.4f} alpha={ALPHA:.4f} steps={PGD_STEPS}\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        imgs = augment(imgs)

        model.eval()
        adv = pgd_attack(model, imgs, lbls, EPS, ALPHA, PGD_STEPS)
        model.train()

        optimizer.zero_grad()
        logits = model(adv)
        loss = criterion(logits, lbls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(lbls)
        correct    += (logits.argmax(1) == lbls).sum().item()
        total      += len(lbls)

    scheduler.step()
    lr_now     = optimizer.param_groups[0]['lr']
    train_loss = total_loss / total
    train_acc  = correct / total

    if epoch % 5 == 0 or epoch == 1:
        clean_acc  = evaluate(model, val_loader, adversarial=False)
        robust_acc = evaluate(model, val_loader, adversarial=True)
        score      = 0.5 * clean_acc + 0.5 * robust_acc
        print(f"Epoch {epoch:3d}/{EPOCHS} | loss={train_loss:.4f} | "
              f"train_adv={train_acc:.3f} | clean={clean_acc:.3f} | "
              f"robust={robust_acc:.3f} | score={score:.3f} | lr={lr_now:.5f}")
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  *** New best score: {score:.4f} — model saved ***")
    else:
        print(f"Epoch {epoch:3d}/{EPOCHS} | loss={train_loss:.4f} | "
              f"train_adv={train_acc:.3f} | lr={lr_now:.5f}")

print(f"\nTraining complete. Best score: {best_score:.4f}")
print(f"Model saved to: {MODEL_PATH}")
