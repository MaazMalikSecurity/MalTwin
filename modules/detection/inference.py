from pathlib import Path

import numpy as np
import torch

from modules.detection.model import MalTwinCNN
from modules.enhancement.augmentor import get_val_transforms


def load_model(model_path: Path, num_classes: int, device: torch.device) -> MalTwinCNN:
    """
    Loads MalTwinCNN from .pt checkpoint file.
    SRS ref: FE-5 of Module 5, SI-3
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = MalTwinCNN(num_classes=num_classes)
    state = torch.load(model_path, map_location=device)
    # Handle both raw state_dict and checkpoint dicts
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_single(
    model: MalTwinCNN,
    img_array: np.ndarray,
    class_names: list,
    device: torch.device,
) -> dict:
    """
    Runs inference on a single image. Returns prediction dict.
    SRS ref: FE-4 of Module 5, REL-1
    """
    img = img_array.astype(np.float32) / 255.0
    transform = get_val_transforms(img_array.shape[0])
    # img shape: (H, W) → (1, H, W) tensor
    tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
    tensor = transform(tensor)                    # normalize
    tensor = tensor.unsqueeze(0).to(device)       # (1, 1, H, W)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    predicted_family = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"

    probabilities = {
        name: float(probs[i]) if i < len(probs) else 0.0
        for i, name in enumerate(class_names)
    }

    return {
        "predicted_family": predicted_family,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def predict_batch(
    model: MalTwinCNN,
    img_arrays: list,
    class_names: list,
    device: torch.device,
) -> list:
    """
    Runs inference on a batch of images.
    Returns list of prediction dicts.
    """
    results = []
    for img_array in img_arrays:
        results.append(predict_single(model, img_array, class_names, device))
    return results
