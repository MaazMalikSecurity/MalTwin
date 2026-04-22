from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.detection.model import MalTwinCNN


def validate(
    model: MalTwinCNN,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple:
    """
    Returns (avg_loss, accuracy) for one pass through loader.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(
    model: MalTwinCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
    best_model_path: Path,
) -> dict:
    """
    Full training loop. Returns history dict.
    SRS ref: FE-2 of Module 5, OE-4, REL-1
    """
    torch.manual_seed(42)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = Path(best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs:03d}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        val_loss, val_acc = validate(model, val_loader, device, criterion)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            checkpoint_path,
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best Val Acc: {best_val_acc:.4f} saved to {best_model_path}")

    return history
