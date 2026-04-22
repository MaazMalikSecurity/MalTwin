import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from modules.detection.model import MalTwinCNN


def evaluate(
    model: MalTwinCNN,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list,
) -> dict:
    """
    Full test-set evaluation.
    SRS ref: FE-3, BO-5
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float(accuracy_score(all_labels, all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    per_class: dict[str, dict] = {}
    for idx, name in enumerate(class_names):
        if idx < len(per_class_precision):
            per_class[name] = {
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "f1": float(per_class_f1[idx]),
            }

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "per_class": per_class,
    }
