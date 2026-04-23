# MalTwin — Phase 5: CLI Scripts
### Agent Instruction Document | `scripts/train.py` + `scripts/evaluate.py` + `scripts/convert_binary.py`

> **Read this entire document before writing a single line of code.**
> All three scripts are fully specified here. Do not add argparse flags, imports,
> print statements, or error handling that is not listed below.

---

## Mandatory Rules (from PRD Section 16)

- **Read `MALTWIN_PRD_COMPLETE.md`** before writing any code.
- Scripts are in the `scripts/` directory at the repo root, **not** inside `modules/`.
- All scripts must be runnable as `python scripts/<name>.py` from the repo root.
- All paths constructed in scripts must use `pathlib.Path`, never string concatenation.
- No script may import from another script — only from `modules/` and `config`.
- `torch.manual_seed()` and `np.random.seed()` are called **at the top of `main()`**, before any data loading.
- Exit codes must be exact: `sys.exit(0)` on success, `sys.exit(1)` for dataset errors, `sys.exit(2)` for training errors.
- Scripts must have no side effects at import time (all logic inside `if __name__ == '__main__': main()`).
- `argparse` is the only CLI parsing library permitted — no `click`, no `typer`.

---

## Phase 5 Scope

Phase 5 implements the three CLI scripts that orchestrate the full ML pipeline. They depend on Phases 1–4 being complete and correct.

### Files to create

| File | Description |
|------|-------------|
| `scripts/__init__.py` | Empty file (makes scripts importable for import-error smoke tests) |
| `scripts/train.py` | Full training pipeline: validate → dataloaders → model → train → evaluate → save outputs |
| `scripts/evaluate.py` | Test-set evaluation only: load model → build test loader → evaluate → print metrics |
| `scripts/convert_binary.py` | Single-file binary-to-image conversion: validate → hash → convert → save PNG |

---

## File 1: `scripts/__init__.py`

```python
# scripts/__init__.py
# Empty — allows `python -c "import scripts.train"` import smoke tests.
```

---

## File 2: `scripts/train.py`

```python
#!/usr/bin/env python3
"""
Full MalTwin training pipeline.

Usage:
    python scripts/train.py [OPTIONS]

Options (all optional — defaults come from config.py / .env):
    --data-dir    PATH   Path to Malimg dataset root     [default: config.DATA_DIR]
    --epochs      INT    Number of training epochs        [default: config.EPOCHS]
    --lr          FLOAT  Learning rate                    [default: config.LR]
    --batch-size  INT    Batch size                       [default: config.BATCH_SIZE]
    --workers     INT    DataLoader worker processes      [default: config.NUM_WORKERS]
    --oversample  STR    Oversampling strategy            [default: config.OVERSAMPLE_STRATEGY]
                         Choices: oversample_minority | sqrt_inverse | uniform
    --no-augment         Disable training augmentation    [flag, default: augmentation ON]
    --seed        INT    Random seed                      [default: config.RANDOM_SEED]

Exit codes:
    0  success
    1  dataset not found or invalid
    2  training or evaluation error

Outputs (written to disk on success):
    models/best_model.pt                   ← best checkpoint by val_acc
    models/checkpoints/epoch_NNN_accX.pt   ← per-epoch checkpoints
    data/processed/class_names.json        ← ordered class name list for dashboard
    data/processed/eval_metrics.json       ← test-set metrics for dashboard KPI cards
    data/processed/confusion_matrix.png    ← confusion matrix heatmap
"""
import argparse
import json
import sys
import random
import torch
import numpy as np
from pathlib import Path


def parse_args() -> argparse.Namespace:
    import config
    parser = argparse.ArgumentParser(
        description="Train MalTwin CNN on the Malimg malware image dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir',   type=str,   default=str(config.DATA_DIR),
                        help='Path to Malimg dataset root directory')
    parser.add_argument('--epochs',     type=int,   default=config.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr',         type=float, default=config.LR,
                        help='Adam learning rate')
    parser.add_argument('--batch-size', type=int,   default=config.BATCH_SIZE,
                        help='Batch size for DataLoaders')
    parser.add_argument('--workers',    type=int,   default=config.NUM_WORKERS,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--oversample', type=str,   default=config.OVERSAMPLE_STRATEGY,
                        choices=['oversample_minority', 'sqrt_inverse', 'uniform'],
                        help='Class oversampling strategy for training loader')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable training augmentation (use val transforms for train)')
    parser.add_argument('--seed',       type=int,   default=config.RANDOM_SEED,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Seed everything ─────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    import config

    # ── 2. Validate dataset ────────────────────────────────────────────────────
    from modules.dataset.preprocessor import validate_dataset_integrity
    print("=" * 55)
    print("MalTwin Training Pipeline")
    print("=" * 55)
    print("\n[1/6] Validating dataset...")
    try:
        report = validate_dataset_integrity(Path(args.data_dir))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Families found:   {len(report['families'])}")
    print(f"  Total samples:    {report['total']}")
    print(f"  Imbalance ratio:  {report['imbalance_ratio']:.1f}x "
          f"({report['max_class']} vs {report['min_class']})")
    if report['corrupt_files']:
        print(f"  WARNING: {len(report['corrupt_files'])} corrupt file(s) found — skipping")

    # ── 3. Build DataLoaders ───────────────────────────────────────────────────
    print("\n[2/6] Building DataLoaders...")
    try:
        from modules.dataset.loader import get_dataloaders
        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            data_dir=Path(args.data_dir),
            img_size=config.IMG_SIZE,
            batch_size=args.batch_size,
            num_workers=args.workers,
            oversample_strategy=args.oversample,
            augment_train=not args.no_augment,
            random_seed=args.seed,
        )
    except Exception as e:
        print(f"ERROR building DataLoaders: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"  Train batches:  {len(train_loader)}")
    print(f"  Val batches:    {len(val_loader)}")
    print(f"  Test batches:   {len(test_loader)}")
    print(f"  Classes:        {len(class_names)}")
    print(f"  Augmentation:   {'OFF (--no-augment)' if args.no_augment else 'ON'}")
    print(f"  Oversample:     {args.oversample}")

    # ── 4. Build model ─────────────────────────────────────────────────────────
    print("\n[3/6] Initialising model...")
    try:
        from modules.detection.model import MalTwinCNN
        model = MalTwinCNN(num_classes=len(class_names)).to(config.DEVICE)
    except Exception as e:
        print(f"ERROR initialising model: {e}", file=sys.stderr)
        sys.exit(2)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture:       MalTwinCNN")
    print(f"  Total parameters:   {total_params:,}")
    print(f"  Trainable params:   {trainable_params:,}")
    print(f"  Device:             {config.DEVICE}")

    # ── 5. Train ───────────────────────────────────────────────────────────────
    print(f"\n[4/6] Training for {args.epochs} epoch(s)...")
    try:
        from modules.detection.trainer import train
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.DEVICE,
            epochs=args.epochs,
            lr=args.lr,
        )
    except Exception as e:
        print(f"ERROR during training: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"\nTraining complete.")
    print(f"  Best val accuracy: {history['best_val_acc']:.4f} at epoch {history['best_epoch']}")
    print(f"  Model saved to:    {config.BEST_MODEL_PATH}")

    # ── 6. Evaluate on test set ────────────────────────────────────────────────
    print("\n[5/6] Evaluating best model on test set...")
    try:
        from modules.detection.inference import load_model
        from modules.detection.evaluator import evaluate, format_metrics_table, plot_confusion_matrix
        best_model = load_model(config.BEST_MODEL_PATH, len(class_names), config.DEVICE)
        metrics = evaluate(best_model, test_loader, config.DEVICE, class_names)
    except Exception as e:
        print(f"ERROR during evaluation: {e}", file=sys.stderr)
        sys.exit(2)

    print(format_metrics_table(metrics, class_names))

    # ── 7. Save eval metrics JSON ──────────────────────────────────────────────
    print("\n[6/6] Saving outputs...")
    try:
        metrics_path = config.PROCESSED_DIR / 'eval_metrics.json'
        # confusion_matrix is np.ndarray — not JSON serialisable; exclude it
        # classification_report is a string — exclude (too verbose for dashboard)
        serialisable = {
            k: v for k, v in metrics.items()
            if k not in ('confusion_matrix', 'per_class', 'classification_report')
        }
        # per_class contains nested dicts with Python floats/ints — safe to serialise
        serialisable['per_class'] = {
            family: dict(stats) for family, stats in metrics['per_class'].items()
        }
        with open(metrics_path, 'w') as f:
            json.dump(serialisable, f, indent=2)
        print(f"  Eval metrics → {metrics_path}")
    except Exception as e:
        # Non-fatal: metrics file is nice-to-have for dashboard
        print(f"  WARNING: Could not save eval metrics: {e}", file=sys.stderr)

    # ── 8. Save confusion matrix PNG ───────────────────────────────────────────
    try:
        cm_path = config.PROCESSED_DIR / 'confusion_matrix.png'
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
        print(f"  Confusion matrix → {cm_path}")
    except Exception as e:
        print(f"  WARNING: Could not save confusion matrix: {e}", file=sys.stderr)

    print("\n" + "=" * 55)
    print("Done!")
    print(f"  Launch dashboard: streamlit run modules/dashboard/app.py")
    print("=" * 55)
    sys.exit(0)


if __name__ == '__main__':
    main()
```

---

## File 3: `scripts/evaluate.py`

```python
#!/usr/bin/env python3
"""
Evaluate best_model.pt on the test split only (no retraining).

Usage:
    python scripts/evaluate.py [OPTIONS]

Options:
    --model-path  PATH   Path to trained model .pt file   [default: config.BEST_MODEL_PATH]
    --data-dir    PATH   Path to Malimg dataset root       [default: config.DATA_DIR]
    --batch-size  INT    Batch size for test DataLoader    [default: config.BATCH_SIZE]
    --workers     INT    DataLoader worker processes       [default: config.NUM_WORKERS]
    --seed        INT    Random seed (affects split)       [default: config.RANDOM_SEED]
    --save-metrics       Save eval_metrics.json to data/processed/ [flag]

Exit codes:
    0  success
    1  model file or dataset not found
    2  evaluation error

Use case:
    Re-evaluate after code changes or hyperparameter review without retraining.
    Produces the same test split as training (same seed and split ratios).
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    import config
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MalTwin model on the Malimg test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model-path',   type=str, default=str(config.BEST_MODEL_PATH),
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--data-dir',     type=str, default=str(config.DATA_DIR),
                        help='Path to Malimg dataset root directory')
    parser.add_argument('--batch-size',   type=int, default=config.BATCH_SIZE,
                        help='Batch size for test DataLoader')
    parser.add_argument('--workers',      type=int, default=config.NUM_WORKERS,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--seed',         type=int, default=config.RANDOM_SEED,
                        help='Random seed (must match training seed for same test split)')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Save eval_metrics.json to data/processed/')
    return parser.parse_args()


def main():
    args = parse_args()
    import config

    model_path = Path(args.model_path)
    data_dir   = Path(args.data_dir)

    # ── 1. Validate model file exists ─────────────────────────────────────────
    if not model_path.exists():
        print(
            f"ERROR: Model file not found: {model_path}\n"
            "Run scripts/train.py first to produce best_model.pt",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 2. Validate dataset exists ────────────────────────────────────────────
    if not data_dir.exists():
        print(
            f"ERROR: Dataset directory not found: {data_dir}\n"
            "Download Malimg from Kaggle and extract to data/malimg/",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 3. Load class names ───────────────────────────────────────────────────
    print("Loading class names...")
    try:
        from modules.dataset.preprocessor import load_class_names
        class_names = load_class_names(config.CLASS_NAMES_PATH)
        print(f"  {len(class_names)} classes loaded from {config.CLASS_NAMES_PATH}")
    except FileNotFoundError:
        # Fall back to scanning the dataset directory
        print("  class_names.json not found — scanning dataset directory...")
        from modules.dataset.preprocessor import encode_labels
        subdirs = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
        label_map = encode_labels(subdirs)
        class_names = sorted(label_map.keys())
        print(f"  {len(class_names)} classes found in dataset")

    # ── 4. Build test DataLoader ──────────────────────────────────────────────
    print("Building test DataLoader...")
    try:
        from modules.dataset.loader import MalimgDataset
        from modules.enhancement.augmentor import get_val_transforms
        from torch.utils.data import DataLoader

        test_ds = MalimgDataset(
            data_dir=data_dir,
            split='test',
            img_size=config.IMG_SIZE,
            transform=get_val_transforms(config.IMG_SIZE),
            random_seed=args.seed,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        print(f"  Test samples:  {len(test_ds)}")
        print(f"  Test batches:  {len(test_loader)}")
    except Exception as e:
        print(f"ERROR building test DataLoader: {e}", file=sys.stderr)
        sys.exit(2)

    # ── 5. Load model ─────────────────────────────────────────────────────────
    print(f"Loading model from {model_path}...")
    try:
        from modules.detection.inference import load_model
        model = load_model(
            model_path=model_path,
            num_classes=len(class_names),
            device=config.DEVICE,
        )
        print(f"  Device: {config.DEVICE}")
    except Exception as e:
        print(f"ERROR loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    try:
        from modules.detection.evaluator import evaluate, format_metrics_table
        metrics = evaluate(model, test_loader, config.DEVICE, class_names)
    except Exception as e:
        print(f"ERROR during evaluation: {e}", file=sys.stderr)
        sys.exit(2)

    # ── 7. Print results ──────────────────────────────────────────────────────
    print(format_metrics_table(metrics, class_names))
    print("\nClassification Report:")
    print(metrics['classification_report'])

    # ── 8. Optionally save metrics ────────────────────────────────────────────
    if args.save_metrics:
        try:
            metrics_path = config.PROCESSED_DIR / 'eval_metrics.json'
            serialisable = {
                k: v for k, v in metrics.items()
                if k not in ('confusion_matrix', 'per_class', 'classification_report')
            }
            serialisable['per_class'] = {
                family: dict(stats) for family, stats in metrics['per_class'].items()
            }
            with open(metrics_path, 'w') as f:
                json.dump(serialisable, f, indent=2)
            print(f"Eval metrics saved to {metrics_path}")
        except Exception as e:
            print(f"WARNING: Could not save eval metrics: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == '__main__':
    main()
```

---

## File 4: `scripts/convert_binary.py`

```python
#!/usr/bin/env python3
"""
Convert a single PE or ELF binary file to a 128×128 grayscale PNG.

Usage:
    python scripts/convert_binary.py --input FILE [--output FILE.png] [--size INT]

Arguments:
    --input   PATH   Path to input binary file (.exe, .dll, ELF)  [required]
    --output  PATH   Path for output PNG file                      [default: <input>.png]
    --size    INT    Output image size (square)                     [default: 128]

Exit codes:
    0  success
    1  input file not found
    2  invalid binary format (not PE or ELF)
    3  conversion or save error

No ML dependencies. This script works without PyTorch or scikit-learn installed.
"""
import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PE/ELF binary to a grayscale PNG image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input',  type=str, required=True,
                        help='Path to input binary file (.exe, .dll, or ELF)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path for output PNG (default: <input_name>.png)')
    parser.add_argument('--size',   type=int, default=128,
                        help='Output image size in pixels (NxN square)')
    return parser.parse_args()


def main():
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix('.png')
    img_size    = args.size

    # ── 1. Verify input file exists ────────────────────────────────────────────
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not input_path.is_file():
        print(f"ERROR: Input path is not a file: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ── 2. Read raw bytes ──────────────────────────────────────────────────────
    print(f"Reading: {input_path}  ({input_path.stat().st_size:,} bytes)")
    file_bytes = input_path.read_bytes()

    # ── 3. Validate binary format ──────────────────────────────────────────────
    from modules.binary_to_image.utils import validate_binary_format, compute_sha256, get_file_metadata
    try:
        file_format = validate_binary_format(file_bytes)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Format:  {file_format}")

    # ── 4. Compute SHA-256 ─────────────────────────────────────────────────────
    sha256 = compute_sha256(file_bytes)
    print(f"SHA-256: {sha256}")

    # ── 5. Print full metadata ─────────────────────────────────────────────────
    meta = get_file_metadata(file_bytes, input_path.name, file_format)
    print(f"Size:    {meta['size_human']}")
    print(f"Time:    {meta['upload_time']}")

    # ── 6. Convert bytes → grayscale array ────────────────────────────────────
    from modules.binary_to_image.converter import BinaryConverter
    try:
        converter  = BinaryConverter(img_size=img_size)
        img_array  = converter.convert(file_bytes)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERROR during conversion: {e}", file=sys.stderr)
        sys.exit(3)

    # ── 7. Save PNG to disk ────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        converter.save(img_array, output_path)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"\nSaved {img_size}x{img_size} grayscale PNG to {output_path}")
    sys.exit(0)


if __name__ == '__main__':
    main()
```

---

## Definition of Done

Phase 5 has no dedicated pytest test file in the PRD — the definition of done is that all three scripts import cleanly and run without errors. Run these checks:

```bash
# ── Smoke test: verify scripts import without errors ──────────────────────────
python -c "import scripts.train"
python -c "import scripts.evaluate"
python -c "import scripts.convert_binary"

# All three should produce no output and exit with code 0.

# ── Verify argparse help works (no dataset needed) ────────────────────────────
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/convert_binary.py --help

# ── Convert a test binary (uses Phase 1 fixtures) ─────────────────────────────
python scripts/convert_binary.py \
    --input tests/fixtures/sample_pe.exe \
    --output /tmp/test_output.png \
    --size 128

# Expected output:
# Reading: tests/fixtures/sample_pe.exe  (1,024 bytes)
# Format:  PE
# SHA-256: <64 hex chars>
# Size:    1.0 KB
# Time:    <ISO 8601 timestamp>
# Saved 128x128 grayscale PNG to /tmp/test_output.png

# ── Verify exit code 1 on missing input ───────────────────────────────────────
python scripts/convert_binary.py --input /tmp/does_not_exist.exe
echo "Exit code: $?"
# Expected: Exit code: 1

# ── Verify exit code 2 on non-binary input ────────────────────────────────────
echo "not a binary" > /tmp/fake.exe
python scripts/convert_binary.py --input /tmp/fake.exe
echo "Exit code: $?"
# Expected: Exit code: 2

# ── Regression check: all earlier tests still pass ────────────────────────────
pytest tests/test_converter.py tests/test_dataset.py \
       tests/test_enhancement.py tests/test_model.py \
       -v -m "not integration"
```

### With the real Malimg dataset (full smoke test)

```bash
# 2-epoch smoke train — fast check that the full pipeline runs end-to-end
python scripts/train.py --epochs 2 --workers 0

# Expected output (abbreviated):
# ======================================================
# MalTwin Training Pipeline
# ======================================================
# [1/6] Validating dataset...
#   Families found:   25
#   Total samples:    9339
#   Imbalance ratio:  36.9x (Allaple.A vs Skintrim.N)
# [2/6] Building DataLoaders...
# [3/6] Initialising model...
# [4/6] Training for 2 epoch(s)...
# [5/6] Evaluating best model on test set...
# [6/6] Saving outputs...
# Done!

# Verify outputs exist
ls -lh models/best_model.pt
ls -lh data/processed/class_names.json
ls -lh data/processed/eval_metrics.json
ls -lh data/processed/confusion_matrix.png

# Evaluate-only script
python scripts/evaluate.py

# Expected: prints the same metrics table as training
```

---

### Checklist

- [ ] `python -c "import scripts.train"` exits with code 0 (no import errors)
- [ ] `python -c "import scripts.evaluate"` exits with code 0
- [ ] `python -c "import scripts.convert_binary"` exits with code 0
- [ ] `python scripts/train.py --help` prints usage without error
- [ ] `python scripts/evaluate.py --help` prints usage without error
- [ ] `python scripts/convert_binary.py --help` prints usage without error
- [ ] `python scripts/convert_binary.py --input tests/fixtures/sample_pe.exe --output /tmp/t.png` exits 0 and creates PNG
- [ ] Missing input file → exit code 1
- [ ] Non-PE/ELF input → exit code 2
- [ ] `scripts/__init__.py` exists (even if empty)
- [ ] All three scripts have `if __name__ == '__main__': main()` guard
- [ ] No script imports from another script
- [ ] `torch.manual_seed()` called in `train.py`'s `main()`, not at module level
- [ ] `sys.exit(0)` at end of each successful `main()`

---

## Common Bugs to Avoid

| Bug | Symptom | Fix |
|-----|---------|-----|
| Logic outside `if __name__ == '__main__'` | Import smoke test triggers side effects | Wrap all logic in `main()`, guard with `__name__` check |
| `sys.exit()` inside `parse_args()` | Argparse `--help` causes unexpected exit codes | Only call `sys.exit()` inside `main()` |
| Missing `scripts/__init__.py` | `python -c "import scripts.train"` raises `ModuleNotFoundError` | Create the empty file |
| `str(path)` used in argparse default before `config` import | `AttributeError` at parse time | Import `config` inside `parse_args()` body, not at module level |
| `evaluate.py` building train loader instead of test loader | Wrong split evaluated | Always pass `split='test'` to `MalimgDataset` in `evaluate.py` |
| `--seed` default hardcoded as `42` instead of `config.RANDOM_SEED` | Seed differs from training if `.env` overrides it | Use `default=config.RANDOM_SEED` in argparse |
| `shuffle=True` on test DataLoader in `evaluate.py` | Non-reproducible evaluation order | `shuffle=False` on test and val loaders always |
| Catching all exceptions around `validate_dataset_integrity` | Dataset not found swallowed silently | Only catch `FileNotFoundError` for missing dataset; re-raise or `sys.exit(1)` |
| `confusion_matrix` key passed to `json.dump` | `TypeError: Object of type ndarray is not JSON serializable` | Exclude `confusion_matrix`, `per_class` raw, and `classification_report` before dumping |

---

*Phase 5 complete → proceed to Phase 6: Dashboard (`modules/dashboard/db.py`, `state.py`, `pages/`, `app.py`).*
