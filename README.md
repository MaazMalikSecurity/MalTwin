# MalTwin — Implementation PRD & Developer Reference

> **Target: 30% implementation** covering Modules 2, 3, 4, and 5 (Binary-to-Image Pipeline, Dataset, Data Enhancement, and CNN Detection) plus the Streamlit Dashboard shell (Module 6) with file upload and prediction screens wired end-to-end.

---

## 0. Quick-Start Scope Decision

The 30% target is defined as:

| Module | Status | Rationale |
|--------|--------|-----------|
| M1 — Digital Twin Simulation | ❌ Deferred | Requires Docker + Mininet infra setup; independent of ML pipeline |
| M2 — Binary-to-Image Conversion | ✅ **Implement** | Core pipeline entry point; zero external deps beyond numpy/opencv |
| M3 — Dataset Collection & Preprocessing | ✅ **Implement** | Required before any model training is possible |
| M4 — Data Enhancement & Balancing | ✅ **Implement** | Needed for training quality; pure Python/torch transforms |
| M5 — Intelligent Malware Detection (CNN) | ✅ **Implement** | Core deliverable; training + inference scripts |
| M6 — Dashboard & Visualization | ✅ **Partial** | Upload screen + detection screen only; DT monitor tab stubbed |
| M7 — Explainable AI (Grad-CAM) | ❌ Deferred | Depends on trained M5 model; add after first training run |
| M8 — Automated Threat Reporting | ❌ Deferred | Depends on M5 + M7 results |

---

## 1. Repository Structure

```
maltwin/
├── README.md                        ← this file
├── requirements.txt                 ← all pip dependencies pinned
├── .env.example                     ← env vars template
├── config.py                        ← central config (paths, hyperparams, constants)
│
├── modules/
│   ├── __init__.py
│   ├── binary_to_image/
│   │   ├── __init__.py
│   │   ├── converter.py             ← BinaryConverter class
│   │   └── utils.py                 ← header validation, sha256, histogram
│   │
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── loader.py                ← MalimgDataset, stratified split
│   │   └── preprocessor.py         ← normalize, encode labels, integrity check
│   │
│   ├── enhancement/
│   │   ├── __init__.py
│   │   ├── augmentor.py             ← AugmentationPipeline (transforms)
│   │   └── balancer.py              ← ClassAwareOversampler
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── model.py                 ← MalTwinCNN architecture (PyTorch)
│   │   ├── trainer.py               ← train(), validate(), save_checkpoint()
│   │   ├── evaluator.py             ← accuracy/precision/recall/F1/confusion matrix
│   │   └── inference.py             ← load_model(), predict_single(), predict_batch()
│   │
│   └── dashboard/
│       ├── __init__.py
│       ├── app.py                   ← Streamlit entry point
│       └── pages/
│           ├── home.py              ← M1 status overview (stubbed)
│           ├── upload.py            ← Binary upload + grayscale visualization
│           └── detection.py         ← Run detection + prediction display
│
├── data/
│   ├── malimg/                      ← Malimg dataset root (one folder per family)
│   │   └── .gitkeep
│   ├── processed/                   ← Preprocessed numpy arrays cached here
│   │   └── .gitkeep
│   └── mitre_ics_mapping.json       ← Local MITRE ATT&CK for ICS reference
│
├── models/
│   ├── checkpoints/                 ← Per-epoch .pt files saved here
│   │   └── .gitkeep
│   └── best_model.pt                ← Best validation accuracy weights
│
├── reports/                         ← Generated PDF/JSON forensic reports (M8)
│   └── .gitkeep
│
├── logs/
│   ├── maltwin.db                   ← SQLite detection event log
│   └── .gitkeep
│
├── scripts/
│   ├── train.py                     ← CLI training entry point
│   ├── evaluate.py                  ← CLI evaluation on test split
│   └── convert_binary.py            ← CLI single-file binary-to-image conversion
│
└── tests/
    ├── test_converter.py
    ├── test_dataset.py
    ├── test_enhancement.py
    ├── test_model.py
    └── fixtures/
        ├── sample.exe               ← small benign PE for unit tests
        └── sample.elf               ← small benign ELF for unit tests
```

---

## 2. Environment & Dependencies

### 2.1 System Requirements
- OS: Ubuntu 22.04 LTS (kernel 5.15+)
- Python: 3.11.x (3.14 per SRS; use 3.11 for library compatibility — adjust once 3.14 stable)
- GPU (optional): NVIDIA GPU with CUDA 12.x, minimum 6 GB VRAM
- RAM: 16 GB minimum
- Disk: 20 GB free for dataset + models (100 GB if running M1 Docker simulation)

### 2.2 `requirements.txt` (exact pins required)

```
# Core ML
torch==2.3.1
torchvision==0.18.1
captum==0.7.0

# Image processing
opencv-python-headless==4.10.0.84
Pillow==10.4.0
numpy==1.26.4

# Dataset / data processing
scikit-learn==1.5.1
imbalanced-learn==0.12.3
pandas==2.2.2

# Dashboard
streamlit==1.37.0
plotly==5.23.0

# Reporting (M8 — install now, implement later)
fpdf2==2.7.9

# Utilities
python-dotenv==1.0.1
tqdm==4.66.5
pytest==8.3.2
```

### 2.3 `.env.example`

```
MALTWIN_DATA_DIR=./data/malimg
MALTWIN_MODEL_DIR=./models
MALTWIN_LOG_DIR=./logs
MALTWIN_REPORTS_DIR=./reports
MALTWIN_IMG_SIZE=128
MALTWIN_BATCH_SIZE=32
MALTWIN_EPOCHS=30
MALTWIN_LR=0.001
MALTWIN_NUM_WORKERS=4
MALTWIN_DEVICE=auto          # auto | cpu | cuda
```

---

## 3. `config.py` — Central Configuration

```python
# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# Paths
DATA_DIR        = Path(os.getenv("MALTWIN_DATA_DIR", BASE_DIR / "data/malimg"))
PROCESSED_DIR   = BASE_DIR / "data/processed"
MODEL_DIR       = Path(os.getenv("MALTWIN_MODEL_DIR", BASE_DIR / "models"))
CHECKPOINT_DIR  = MODEL_DIR / "checkpoints"
LOG_DIR         = Path(os.getenv("MALTWIN_LOG_DIR", BASE_DIR / "logs"))
REPORTS_DIR     = Path(os.getenv("MALTWIN_REPORTS_DIR", BASE_DIR / "reports"))
MITRE_JSON_PATH = BASE_DIR / "data/mitre_ics_mapping.json"
DB_PATH         = LOG_DIR / "maltwin.db"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pt"

# Image settings
IMG_SIZE        = int(os.getenv("MALTWIN_IMG_SIZE", 128))   # 128x128 per SRS FR3.2

# Training hyperparameters
BATCH_SIZE      = int(os.getenv("MALTWIN_BATCH_SIZE", 32))
EPOCHS          = int(os.getenv("MALTWIN_EPOCHS", 30))
LR              = float(os.getenv("MALTWIN_LR", 0.001))
NUM_WORKERS     = int(os.getenv("MALTWIN_NUM_WORKERS", 4))
TRAIN_RATIO     = 0.7
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15   # must sum to 1.0

# Device
import torch
_device_env = os.getenv("MALTWIN_DEVICE", "auto")
if _device_env == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(_device_env)

# Binary upload limits
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per SRS FR3.1
ACCEPTED_EXTENSIONS = {".exe", ".dll", ".elf", ""}  # "" covers extensionless ELF

# Confidence thresholds for UI color coding (SRS FR5.2)
CONFIDENCE_GREEN = 0.80
CONFIDENCE_AMBER = 0.50
```

---

## 4. Module 2 — Binary-to-Image Conversion

### 4.1 `modules/binary_to_image/utils.py`

Implement these functions exactly:

```python
def validate_binary_format(file_bytes: bytes) -> str:
    """
    Returns 'PE', 'ELF', or raises ValueError.
    PE magic: first 2 bytes == b'MZ'
    ELF magic: first 4 bytes == b'\x7fELF'
    SRS ref: UC-01 Alternate Flow A1, FR-B1
    """

def compute_sha256(file_bytes: bytes) -> str:
    """
    Returns lowercase hex SHA-256 digest.
    Uses hashlib.sha256 only — no external services.
    SRS ref: SRS FR3.3, SEC-4, CON-9
    """

def compute_pixel_histogram(img_array: np.ndarray) -> dict:
    """
    Returns dict with keys 'bins' (list 0-255) and 'counts' (list of int).
    256 bins exactly, one per byte value.
    SRS ref: SRS FR3.4
    """
```

### 4.2 `modules/binary_to_image/converter.py`

```python
class BinaryConverter:
    """
    Converts raw PE/ELF binary bytes to 128x128 grayscale PNG.

    Constructor args:
        img_size (int): output image side length in pixels (default from config.IMG_SIZE)

    Public methods:
        convert(file_bytes: bytes) -> np.ndarray
            - Read bytes as uint8 array
            - Calculate width = int(math.sqrt(len(bytes)))
            - Trim array so length is divisible by width (trim tail bytes)
            - Reshape to (height, width) 2D array
            - Resize to (img_size, img_size) using cv2.INTER_LINEAR (bilinear interpolation)
            - Return as uint8 numpy array shape (img_size, img_size)
            - SRS ref: FE-2, FE-3 of Module 2

        to_png_bytes(img_array: np.ndarray) -> bytes
            - Encode grayscale array to PNG bytes using cv2.imencode
            - Return bytes for Streamlit st.image() display

        save(img_array: np.ndarray, output_path: Path) -> None
            - Save PNG to disk using cv2.imwrite

    Error handling:
        - If file_bytes is empty or len < 4, raise ValueError("Binary file is empty or too small")
        - All exceptions propagate to caller (dashboard catches and displays user-friendly message)
    """
```

---

## 5. Module 3 — Dataset Collection & Preprocessing

### 5.1 Dataset: Malimg

**Source**: https://www.kaggle.com/datasets/rraftogianou/malimg-dataset  
**Structure expected** (already in grayscale PNG, one folder per malware family):
```
data/malimg/
├── Adialer.C/          ← 122 samples
├── Agent.FYI/          ← 116 samples
├── Allaple.A/          ← 2949 samples  ← heavily imbalanced
├── Allaple.L/          ← 1591 samples
... (25 families total)
```

**No internet download in code** — user downloads manually and places at `DATA_DIR`. Code only reads from filesystem.

### 5.2 `modules/dataset/preprocessor.py`

```python
def validate_dataset_integrity(data_dir: Path) -> dict:
    """
    Walks data_dir subdirectories.
    Returns {
        'families': list[str],          # sorted list of class folder names
        'counts': dict[str, int],       # samples per family
        'total': int,
        'corrupt_files': list[Path],    # files that fail cv2.imread()
        'duplicate_hashes': list[str],  # SHA-256 duplicates found
    }
    Raises FileNotFoundError if data_dir does not exist.
    SRS ref: FE-4 of Module 3
    """

def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Converts uint8 [0,255] to float32 [0.0, 1.0].
    SRS ref: FE-2 of Module 3
    """

def encode_labels(families: list[str]) -> dict[str, int]:
    """
    Returns {family_name: integer_class_index} sorted alphabetically.
    Deterministic — same input always produces same mapping.
    """
```

### 5.3 `modules/dataset/loader.py`

```python
class MalimgDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping the Malimg directory structure.

    Constructor args:
        data_dir (Path): root of malimg folder
        img_size (int): resize target (default config.IMG_SIZE)
        transform (callable): optional torchvision transform
        split (str): 'train' | 'val' | 'test'
        train_ratio / val_ratio / test_ratio: split fractions

    Internals:
        - Uses sklearn.model_selection.train_test_split with stratify=labels
          to produce reproducible splits (random_state=42)
        - Stores list of (file_path, label_int) tuples for the requested split
        - __getitem__ reads PNG with cv2.imread(..., cv2.IMREAD_GRAYSCALE),
          resizes to (img_size, img_size), normalizes to [0,1] float32,
          adds channel dim to get shape (1, img_size, img_size),
          applies transform if provided, returns (tensor, label_int)

    Properties:
        class_names: list[str]   # ordered list matching label integers
        class_counts: dict[str, int]  # samples per class IN THIS SPLIT

    SRS ref: FE-1, FE-2, FE-3 of Module 3
    """

def get_dataloaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    augment_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    train_loader uses AugmentationPipeline transforms (see Module 4).
    val_loader and test_loader use only normalize transform.
    SRS ref: FE-3 of Module 3
    """
```

---

## 6. Module 4 — Data Enhancement & Balancing

### 6.1 `modules/enhancement/augmentor.py`

```python
def get_train_transforms(img_size: int) -> torchvision.transforms.Compose:
    """
    Returns a Compose pipeline for TRAINING data only. Contains:
        1. torchvision.transforms.RandomRotation(degrees=15)
           SRS ref: FE-1 — up to ±15 degrees
        2. torchvision.transforms.RandomHorizontalFlip(p=0.5)
           SRS ref: FE-1
        3. torchvision.transforms.RandomVerticalFlip(p=0.5)
           SRS ref: FE-1
        4. torchvision.transforms.ColorJitter(brightness=0.2)
           SRS ref: FE-1 — brightness adjustment
        5. GaussianNoise (custom transform, see below)
           SRS ref: FE-2
        6. torchvision.transforms.ToTensor()
        7. torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    NOTE: Input images are single-channel (grayscale). Normalize uses 1-channel params.
    """

class GaussianNoise:
    """
    Custom torchvision-compatible transform.
    Applied to PIL Image or numpy array.
    Adds Gaussian noise with mean=0, std sampled uniformly from [0.01, 0.05].
    SRS ref: FE-2 of Module 4
    __call__(self, tensor: torch.Tensor) -> torch.Tensor
    Clamps output to [0.0, 1.0] after noise injection.
    """

def get_val_transforms(img_size: int) -> torchvision.transforms.Compose:
    """
    Returns Compose for val/test — NO augmentation, only:
        1. torchvision.transforms.ToTensor()
        2. torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    """
```

### 6.2 `modules/enhancement/balancer.py`

```python
class ClassAwareOversampler:
    """
    Wraps torch.utils.data.WeightedRandomSampler.

    Constructor args:
        dataset (MalimgDataset): the TRAINING split dataset
        strategy (str): 'oversample_minority' | 'sqrt_inverse' | 'uniform'
            - 'oversample_minority': weight = 1/class_count (standard inverse freq)
            - 'sqrt_inverse': weight = 1/sqrt(class_count) (softer balancing)
            - 'uniform': all classes equal weight regardless of size

    get_sampler() -> WeightedRandomSampler
        Returns sampler that draws len(dataset) samples per epoch with replacement.
        This replaces shuffle=True in the DataLoader.
        SRS ref: FE-3 of Module 4

    Intended usage in get_dataloaders():
        sampler = ClassAwareOversampler(train_dataset).get_sampler()
        DataLoader(train_dataset, batch_size=..., sampler=sampler)
    """
```

---

## 7. Module 5 — Intelligent Malware Detection

### 7.1 `modules/detection/model.py`

```python
class MalTwinCNN(nn.Module):
    """
    Custom CNN for grayscale malware image classification.

    Input shape: (batch, 1, 128, 128)  — single-channel grayscale
    Output shape: (batch, num_classes)  — raw logits

    Architecture (implement exactly):
    ┌─────────────────────────────────────────────┐
    │ Block 1: Conv2d(1→32, 3x3, pad=1)           │
    │          BatchNorm2d(32)                     │
    │          ReLU                                │
    │          Conv2d(32→32, 3x3, pad=1)           │
    │          BatchNorm2d(32)                     │
    │          ReLU                                │
    │          MaxPool2d(2x2) → (batch,32,64,64)   │
    │          Dropout2d(0.25)                     │
    ├─────────────────────────────────────────────┤
    │ Block 2: Conv2d(32→64, 3x3, pad=1)           │
    │          BatchNorm2d(64)                     │
    │          ReLU                                │
    │          Conv2d(64→64, 3x3, pad=1)           │
    │          BatchNorm2d(64)                     │
    │          ReLU                                │
    │          MaxPool2d(2x2) → (batch,64,32,32)   │
    │          Dropout2d(0.25)                     │
    ├─────────────────────────────────────────────┤
    │ Block 3: Conv2d(64→128, 3x3, pad=1)          │
    │          BatchNorm2d(128)                    │
    │          ReLU                                │
    │          Conv2d(128→128, 3x3, pad=1)         │
    │          BatchNorm2d(128)                    │
    │          ReLU                                │
    │          MaxPool2d(2x2) → (batch,128,16,16)  │
    │          Dropout2d(0.25)                     │
    ├─────────────────────────────────────────────┤
    │ Classifier:                                  │
    │   AdaptiveAvgPool2d(4,4) → (batch,128,4,4)  │
    │   Flatten → (batch, 2048)                    │
    │   Linear(2048→512)                           │
    │   ReLU                                       │
    │   Dropout(0.5)                               │
    │   Linear(512→num_classes)                    │
    └─────────────────────────────────────────────┘

    The final convolutional layer for Grad-CAM (Module 7) is Block 3's second Conv2d.
    Store reference as self.gradcam_layer for easy hook registration later.

    Constructor args:
        num_classes (int): number of malware families (25 for Malimg)

    SRS ref: FE-1 of Module 5
    """
```

### 7.2 `modules/detection/trainer.py`

```python
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
    Full training loop. Returns history dict:
    {
        'train_loss': list[float],   # per epoch
        'train_acc':  list[float],
        'val_loss':   list[float],
        'val_acc':    list[float],
    }

    Implementation requirements:
    - Optimizer: torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    - Loss: nn.CrossEntropyLoss()
    - LR Scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau(
          optimizer, mode='max', factor=0.5, patience=5
      )
    - Save checkpoint after every epoch to checkpoint_dir/epoch_{n:03d}.pt
      using torch.save({'epoch': n, 'model_state': ..., 'optimizer_state': ..., 'val_acc': ...})
    - Track best val_acc; if current epoch beats best, save to best_model_path
    - Print per-epoch tqdm progress bar showing loss and acc
    - SRS ref: FE-2 of Module 5, OE-4 (GPU support), REL-1 (deterministic)
    """

def validate(
    model: MalTwinCNN,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Returns (avg_loss, accuracy) for one pass through loader.
    Sets model.eval(), uses torch.no_grad().
    """
```

### 7.3 `modules/detection/evaluator.py`

```python
def evaluate(
    model: MalTwinCNN,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """
    Full test-set evaluation. Returns:
    {
        'accuracy':  float,
        'precision': float,   # macro average
        'recall':    float,   # macro average
        'f1':        float,   # macro average
        'confusion_matrix': np.ndarray shape (num_classes, num_classes),
        'per_class': {
            family_name: {'precision': float, 'recall': float, 'f1': float}
        }
    }
    Uses sklearn.metrics: accuracy_score, precision_recall_fscore_support,
    confusion_matrix.
    SRS ref: FE-3, BO-5
    """
```

### 7.4 `modules/detection/inference.py`

```python
def load_model(model_path: Path, num_classes: int, device: torch.device) -> MalTwinCNN:
    """
    Loads MalTwinCNN from .pt checkpoint file.
    Handles state_dict loading with map_location=device.
    Raises FileNotFoundError if model_path does not exist.
    Sets model.eval() before returning.
    SRS ref: FE-5 of Module 5, SI-3
    """

def predict_single(
    model: MalTwinCNN,
    img_array: np.ndarray,   # raw 128x128 uint8 grayscale from BinaryConverter
    class_names: list[str],
    device: torch.device,
) -> dict:
    """
    Runs inference on a single image. Returns:
    {
        'predicted_family': str,          # top-1 class name
        'confidence': float,              # top-1 softmax probability [0,1]
        'probabilities': dict[str, float] # all classes: {name: prob}
    }
    Steps:
    1. Normalize img_array to [0,1] float32
    2. Apply val_transforms (Normalize mean=0.5 std=0.5)
    3. Add batch + channel dims → (1,1,128,128)
    4. model(tensor) → logits
    5. torch.softmax(logits, dim=1) → probabilities
    6. argmax → predicted class
    SRS ref: FE-4 of Module 5, REL-1
    """
```

---

## 8. Module 6 — Dashboard (Partial Implementation)

### 8.1 `modules/dashboard/app.py`

```python
"""
Streamlit entry point.
Run with: streamlit run modules/dashboard/app.py --server.port 8501

Structure:
- st.set_page_config(page_title="MalTwin", layout="wide", initial_sidebar_state="expanded")
- Sidebar with st.radio or st.selectbox for page navigation:
    Options: ["Dashboard", "Binary Upload", "Malware Detection", "Digital Twin (Coming Soon)"]
- Renders selected page by calling the corresponding page module function
- Loads class_names from data/processed/class_names.json on startup
  (written by dataset loader during first run)
- Loads model from BEST_MODEL_PATH if it exists; stores in st.session_state['model']

Session state keys used across pages:
    st.session_state['model']           MalTwinCNN or None
    st.session_state['class_names']     list[str] or None
    st.session_state['img_array']       np.ndarray(128,128) or None
    st.session_state['file_meta']       dict or None (name, size, format, sha256)
    st.session_state['detection_result'] dict or None (from predict_single)
"""
```

### 8.2 `modules/dashboard/pages/upload.py`

```python
"""
Binary Upload & Visualization page — implements SRS Mockup M3.

render() function must:

1. st.file_uploader("Upload Binary File", type=["exe", "dll"])
   - Accept .exe and .dll via type param; ELF binaries have no extension so also
     add a note that extensionless files can be renamed .elf for upload
   - After upload, check file size: if > MAX_UPLOAD_BYTES → st.error() + return

2. Call validate_binary_format(file_bytes) → catch ValueError → st.error() + return

3. Call BinaryConverter().convert(file_bytes) → store in session_state['img_array']

4. Call compute_sha256(file_bytes)

5. Store in session_state['file_meta']:
   {
       'name': uploaded_file.name,
       'size_bytes': len(file_bytes),
       'format': 'PE' or 'ELF',
       'sha256': sha256_hex,
   }

6. Layout: two columns
   Left col: display grayscale image with st.image(img_bytes, caption="128×128 grayscale")
   Right col:
       - File metadata table (st.table or st.dataframe)
       - Pixel intensity histogram using plotly bar chart (256 bins)
         x-axis: byte value 0–255, y-axis: pixel count
         SRS ref: FR3.4

7. st.success("File processed. Navigate to Malware Detection to run analysis.")

Error messages must be plain English per SRS USE-3:
   Format: "Error: [what went wrong]. Cause: [why]. Action: [what to do]."
"""
```

### 8.3 `modules/dashboard/pages/detection.py`

```python
"""
Malware Detection & Prediction page — implements SRS Mockup M5.

render() function must:

1. Guard: if session_state['img_array'] is None → st.warning("Please upload a binary file first.") + return
   Guard: if session_state['model'] is None → st.warning("No trained model found. Train the model first (see scripts/train.py).") + return

2. Display the stored grayscale image (thumbnail, left column)
   Display file metadata summary (right column)

3. st.button("Run Detection")
   On click:
   a. Call predict_single(model, img_array, class_names, device)
   b. Store result in session_state['detection_result']
   c. Log to SQLite (call log_detection_event() from a thin db.py helper)

4. If session_state['detection_result'] is not None, display results:

   a. Predicted family label in large bold text
      Color based on confidence:
        >= CONFIDENCE_GREEN  → st.success()
        >= CONFIDENCE_AMBER  → st.warning() with "Low confidence — verify manually"
        <  CONFIDENCE_AMBER  → st.error()   with "Very low confidence — manual review required"
      SRS ref: FR5.2

   b. Confidence bar: st.progress(int(confidence * 100))
      Show numeric percentage label next to bar

   c. Per-class probability chart:
      Horizontal bar chart via plotly (all classes, sorted descending by prob)
      Show ALL classes even if prob is 0.0
      SRS ref: FR5.3

   d. MITRE ATT&CK section:
      Load mitre_ics_mapping.json, look up predicted_family key
      If found: display tactics and techniques as st.info() blocks
      If not found: st.info("MITRE ATT&CK mapping not available for this family.")
      SRS ref: FR5.5

   e. XAI Heatmap toggle (STUB for now):
      st.checkbox("Generate Grad-CAM Heatmap (requires trained model)")
      If checked: st.info("Grad-CAM XAI will be available in the next implementation phase.")
      SRS ref: FR5.4

   f. Report download buttons (STUB for now):
      st.download_button disabled with label "Download PDF Report (Coming Soon)"
      SRS ref: FR5.6
"""
```

---

## 9. CLI Scripts

### 9.1 `scripts/train.py`

```python
"""
CLI entry point for training.
Usage: python scripts/train.py [--epochs N] [--lr LR] [--batch-size B] [--data-dir PATH]

Steps:
1. Parse args (argparse); defaults from config.py
2. Call validate_dataset_integrity(DATA_DIR) → print summary table
3. Call get_dataloaders(DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, augment_train=True)
4. Instantiate MalTwinCNN(num_classes=len(class_names)).to(DEVICE)
5. Call train(...) → print final best val_acc
6. Call evaluate(model, test_loader, DEVICE, class_names) → print full metrics table
7. Save class_names to data/processed/class_names.json (needed by dashboard)

Expected output format:
Epoch 001/030 | Train Loss: 2.3412 | Train Acc: 0.4823 | Val Loss: 2.1034 | Val Acc: 0.5312
...
Best Val Acc: 0.9421 saved to models/best_model.pt
=== TEST SET EVALUATION ===
Accuracy:  0.9387
Precision: 0.9412 (macro)
Recall:    0.9371 (macro)
F1:        0.9389 (macro)
"""
```

### 9.2 `scripts/convert_binary.py`

```python
"""
CLI single-file conversion utility.
Usage: python scripts/convert_binary.py --input path/to/file.exe --output path/to/out.png

Steps:
1. Read file bytes
2. validate_binary_format → print "Detected format: PE/ELF"
3. compute_sha256 → print hash
4. BinaryConverter().convert() → save PNG
5. Print: "Saved 128x128 grayscale image to {output}"
"""
```

### 9.3 `scripts/evaluate.py`

```python
"""
CLI evaluation on test split only (no retraining).
Usage: python scripts/evaluate.py [--model-path PATH] [--data-dir PATH]

Loads best_model.pt, runs evaluate() on test split, prints metrics + per-class table.
"""
```

---

## 10. Database Helper

### `modules/dashboard/db.py`

```python
"""
SQLite event logging helper.

Schema (create on first import if not exists):

CREATE TABLE IF NOT EXISTS detection_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,         -- ISO 8601 format
    file_name   TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    file_format TEXT NOT NULL,         -- 'PE' or 'ELF'
    file_size   INTEGER NOT NULL,
    predicted_family TEXT NOT NULL,
    confidence  REAL NOT NULL,
    device_used TEXT NOT NULL          -- 'cpu' or 'cuda'
);

Functions to implement:
    init_db(db_path: Path) -> None
        Creates DB file and table if not exists.
        Enables WAL mode: conn.execute("PRAGMA journal_mode=WAL")
        SRS ref: REL-4

    log_detection_event(db_path, file_name, sha256, file_format,
                        file_size, predicted_family, confidence, device) -> None
        Inserts one row. timestamp = datetime.utcnow().isoformat()
        On IntegrityError: retry once. On second failure: log to stderr, do NOT raise.
        SRS ref: FR-B3, FR5 event response

    get_recent_events(db_path, limit=5) -> list[dict]
        Returns last `limit` rows ordered by id DESC.
        Each dict has all column names as keys.
        SRS ref: FR1.4

WAL mode and 600 permission enforcement:
    After init_db(), caller should: os.chmod(db_path, 0o600)
    SRS ref: SEC-3, REL-4
"""
```

---

## 11. `data/mitre_ics_mapping.json` Seed File

Create this file manually with at minimum the 5 most common Malimg families mapped to MITRE ATT&CK for ICS. Expand as needed.

```json
{
  "Allaple.A": {
    "tactics": ["Lateral Movement", "Impact"],
    "techniques": [
      {"id": "T0812", "name": "Default Credentials"},
      {"id": "T0882", "name": "Theft of Operational Information"}
    ]
  },
  "Yuner.A": {
    "tactics": ["Execution", "Persistence"],
    "techniques": [
      {"id": "T0807", "name": "Command-Line Interface"},
      {"id": "T0839", "name": "Module Firmware"}
    ]
  },
  "Instantaccess": {
    "tactics": ["Collection"],
    "techniques": [
      {"id": "T0802", "name": "Automated Collection"}
    ]
  },
  "Swizzor.gen!E": {
    "tactics": ["Defense Evasion"],
    "techniques": [
      {"id": "T0858", "name": "Change Operating Mode"}
    ]
  },
  "VB.AT": {
    "tactics": ["Execution"],
    "techniques": [
      {"id": "T0871", "name": "Execution through API"}
    ]
  }
}
```

---

## 12. Testing Requirements

Each test file must have at minimum these test cases:

### `tests/test_converter.py`
- `test_pe_validation_accepts_valid_pe` — use a real MZ-header bytes prefix
- `test_elf_validation_accepts_valid_elf` — use `\x7fELF` prefix
- `test_invalid_format_raises` — random bytes
- `test_output_shape_is_128x128` — convert known binary, assert shape
- `test_sha256_deterministic` — same input → same hash twice
- `test_empty_bytes_raises_value_error`

### `tests/test_dataset.py`
- `test_validate_integrity_detects_missing_dir`
- `test_split_ratios_sum_to_one`
- `test_stratified_split_no_class_missing_from_val`
- `test_getitem_returns_correct_tensor_shape` — shape must be (1, 128, 128)
- `test_label_encoding_deterministic`

### `tests/test_enhancement.py`
- `test_gaussian_noise_clamps_output_to_01`
- `test_augmented_tensor_same_shape_as_input`
- `test_oversampler_returns_weighted_random_sampler`

### `tests/test_model.py`
- `test_forward_pass_output_shape` — (4, num_classes) for batch=4
- `test_model_parameters_count_reasonable` — sanity check > 1M params
- `test_predict_single_returns_valid_confidence` — confidence in [0,1]
- `test_predict_single_probabilities_sum_to_one`

---

## 13. Implementation Order (Sequential)

Follow this exact order to avoid import or dependency issues:

```
Phase 1 (Foundation — no torch needed yet):
  1. config.py
  2. modules/binary_to_image/utils.py
  3. modules/binary_to_image/converter.py
  4. tests/test_converter.py  ← verify before moving on

Phase 2 (Dataset):
  5. modules/dataset/preprocessor.py
  6. modules/dataset/loader.py
  7. tests/test_dataset.py    ← requires Malimg dataset downloaded

Phase 3 (Enhancement):
  8. modules/enhancement/augmentor.py
  9. modules/enhancement/balancer.py
  10. tests/test_enhancement.py

Phase 4 (Detection):
  11. modules/detection/model.py
  12. modules/detection/trainer.py
  13. modules/detection/evaluator.py
  14. modules/detection/inference.py
  15. tests/test_model.py
  16. scripts/train.py        ← run full training here

Phase 5 (Dashboard):
  17. modules/dashboard/db.py
  18. modules/dashboard/pages/upload.py
  19. modules/dashboard/pages/detection.py
  20. modules/dashboard/app.py
```

---

## 14. Known Constraints for Coding Agents

- **No `form` tags** anywhere in Streamlit code — use `st.button`, `st.file_uploader`, `st.checkbox` directly.
- **No external API calls** — all computation is local. No VirusTotal, no hash lookup services.
- **Malware never touches host filesystem** — in the dashboard, uploaded bytes are read into memory only, never written to disk outside of the Docker container boundary (M1 scope).
- **Single-channel images throughout** — all tensors are `(batch, 1, H, W)`, never RGB. `cv2.IMREAD_GRAYSCALE` flag must always be used.
- **Reproducibility** — `random_state=42` in all sklearn calls, `torch.manual_seed(42)` at start of `train()`.
- **`class_names.json`** must be written to `data/processed/` by `scripts/train.py` before the dashboard can load. If it doesn't exist, dashboard shows a warning.
- **`best_model.pt`** must exist before the detection page is functional. Dashboard gracefully stubs the page if not found.
- **WAL mode** must be set on every SQLite connection open (not just init), because multiple Streamlit threads may write concurrently.
- Do **not** import `modules.dashboard` from training scripts — circular import risk.
- The `GaussianNoise` transform operates on `torch.Tensor` post `ToTensor()`, not PIL images.

---

## 15. How to Run After Full Phase 5 Implementation

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Place Malimg dataset
# Download from Kaggle and extract to: data/malimg/

# 3. Train
python scripts/train.py --epochs 30

# 4. Launch dashboard
streamlit run modules/dashboard/app.py --server.port 8501
# Open http://localhost:8501
```
