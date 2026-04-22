"""
CLI evaluation on test split only (no retraining).
Usage: python scripts/evaluate.py [--model-path PATH] [--data-dir PATH]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Evaluate MalTwinCNN on test split")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .pt model file")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to Malimg dataset root")
    args = parser.parse_args()

    from pathlib import Path
    import config
    from modules.dataset.loader import get_dataloaders
    from modules.detection.inference import load_model
    from modules.detection.evaluator import evaluate

    model_path = Path(args.model_path) if args.model_path else config.BEST_MODEL_PATH
    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR

    print(f"Loading data from: {data_dir}")
    _, _, test_loader = get_dataloaders(
        data_dir=data_dir,
        img_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        augment_train=False,
    )

    class_names = test_loader.dataset.class_names
    num_classes = len(class_names)

    print(f"Loading model from: {model_path}")
    model = load_model(model_path, num_classes, config.DEVICE)

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
