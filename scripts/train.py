"""
CLI training entry point.
Usage: python scripts/train.py [--epochs N] [--lr LR] [--batch-size B] [--data-dir PATH]
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def main():
    parser = argparse.ArgumentParser(description="Train MalTwinCNN")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--data-dir", type=str, default=str(config.DATA_DIR))
    args = parser.parse_args()

    from pathlib import Path
    from modules.dataset.preprocessor import validate_dataset_integrity
    from modules.dataset.loader import get_dataloaders
    from modules.detection.model import MalTwinCNN
    from modules.detection.trainer import train
    from modules.detection.evaluator import evaluate

    data_dir = Path(args.data_dir)

    print("=== Validating Dataset ===")
    integrity = validate_dataset_integrity(data_dir)
    print(f"Families ({len(integrity['families'])}):")
    for family in integrity["families"]:
        print(f"  {family}: {integrity['counts'][family]} samples")
    print(f"Total: {integrity['total']} samples")
    if integrity["corrupt_files"]:
        print(f"WARNING: {len(integrity['corrupt_files'])} corrupt files found")
    if integrity["duplicate_hashes"]:
        print(f"WARNING: {len(integrity['duplicate_hashes'])} duplicate hashes found")

    print("\n=== Loading Data ===")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        img_size=config.IMG_SIZE,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        augment_train=True,
    )

    class_names = train_loader.dataset.class_names
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Save class_names.json
    class_names_path = config.PROCESSED_DIR / "class_names.json"
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_names_path}")

    print("\n=== Training ===")
    model = MalTwinCNN(num_classes=num_classes).to(config.DEVICE)
    print(f"Device: {config.DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=config.CHECKPOINT_DIR,
        best_model_path=config.BEST_MODEL_PATH,
    )

    best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
    print(f"\nBest Val Acc: {best_val_acc:.4f} saved to {config.BEST_MODEL_PATH}")

    print("\n=== TEST SET EVALUATION ===")
    metrics = evaluate(model, test_loader, config.DEVICE, class_names)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (macro)")
    print(f"Recall:    {metrics['recall']:.4f} (macro)")
    print(f"F1:        {metrics['f1']:.4f} (macro)")

    print("\nPer-class metrics:")
    for name, m in metrics["per_class"].items():
        print(f"  {name:30s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
