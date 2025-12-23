"""
I-JEPA Linear Probing Evaluation on CIFAR-100

This script evaluates a pretrained I-JEPA encoder using
LINEAR PROBING on CIFAR-100.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse

from torchvision import transforms
from torchvision.datasets import CIFAR100

from src.help.schedulers import init_model


# -------------------------------------------------------
# Linear Probe Model
# -------------------------------------------------------
class LinearProbe(nn.Module):
    """Linear classifier on top of a frozen encoder"""
    def __init__(self, encoder, num_classes=100, embed_dim=192):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)           # [B, N, D]
            feats = feats.mean(dim=1)         # Global average pooling

        logits = self.classifier(feats)
        return logits


# -------------------------------------------------------
# Evaluation Function
# -------------------------------------------------------
def evaluate_linear_probe(
    encoder_path,
    config_path,
    num_classes=100,
    embed_dim=192,
    batch_size=256,
    num_epochs=90,
    lr=0.1,
    device="cuda",
):
    print("=" * 80)
    print("I-JEPA LINEAR PROBING ON CIFAR-100")
    print("=" * 80)

    # ---------------------------------------------------
    # Load config
    # ---------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    patch_size = config["mask"]["patch_size"]
    crop_size = config["data"]["crop_size"]
    model_name = config["meta"]["model_name"]

    # ---------------------------------------------------
    # Load encoder
    # ---------------------------------------------------
    print(f"\nLoading encoder checkpoint: {encoder_path}")
    checkpoint = torch.load(encoder_path, map_location=device)

    encoder, _ = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=config["meta"]["pred_depth"],
        pred_emb_dim=config["meta"]["pred_emb_dim"],
        model_name=model_name,
    )

    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()

    print(f"Loaded encoder from epoch {checkpoint.get('epoch', 'N/A')}")

    # ---------------------------------------------------
    # Linear probe
    # ---------------------------------------------------
    model = LinearProbe(
        encoder=encoder,
        num_classes=num_classes,
        embed_dim=embed_dim,
    ).to(device)

    # ---------------------------------------------------
    # CIFAR-100 Transforms
    # ---------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2673, 0.2564, 0.2762],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2673, 0.2564, 0.2762],
        ),
    ])

    # ---------------------------------------------------
    # CIFAR-100 Datasets
    # ---------------------------------------------------
    train_dataset = CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset = CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # ---------------------------------------------------
    # Optimizer & Scheduler
    # ---------------------------------------------------
    optimizer = torch.optim.SGD(
        model.classifier.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------
    # Training Loop
    # ---------------------------------------------------
    best_acc = 0.0
    print("\nTraining linear probe...")
    print("=" * 80)

    for epoch in range(num_epochs):
        # -------------------------------
        # Train
        # -------------------------------
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{running_loss / (pbar.n + 1):.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        scheduler.step()

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

        val_acc = 100.0 * correct / total
        print(f"\nEpoch {epoch+1}: Val Acc = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            print("✓ New best accuracy")

    print("=" * 80)
    print(f"BEST CIFAR-100 LINEAR PROBE ACCURACY: {best_acc:.2f}%")
    print("=" * 80)

    return best_acc


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("I-JEPA Linear Probing on CIFAR-100")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    evaluate_linear_probe(
        encoder_path=args.checkpoint,
        config_path=args.config,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
