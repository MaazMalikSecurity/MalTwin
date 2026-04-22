"""
Tests for Module 5 — Intelligent Malware Detection (CNN).
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.detection.model import MalTwinCNN
from modules.detection.inference import predict_single


NUM_CLASSES = 25


def test_forward_pass_output_shape():
    """MalTwinCNN forward pass produces correct output shape (batch, num_classes)."""
    model = MalTwinCNN(num_classes=NUM_CLASSES)
    model.eval()
    x = torch.randn(4, 1, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_model_parameters_count_reasonable():
    """MalTwinCNN has more than 1 million parameters (sanity check)."""
    model = MalTwinCNN(num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 1_000_000, f"Expected > 1M params, got {total_params:,}"


def test_gradcam_layer_is_conv2d():
    """gradcam_layer attribute exists and is a Conv2d layer."""
    import torch.nn as nn
    model = MalTwinCNN(num_classes=NUM_CLASSES)
    assert hasattr(model, "gradcam_layer")
    assert isinstance(model.gradcam_layer, nn.Conv2d)


def test_predict_single_returns_valid_confidence():
    """predict_single returns a confidence value in [0, 1]."""
    model = MalTwinCNN(num_classes=NUM_CLASSES)
    model.eval()
    img_array = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    class_names = [f"Family_{i}" for i in range(NUM_CLASSES)]
    device = torch.device("cpu")
    result = predict_single(model, img_array, class_names, device)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_single_probabilities_sum_to_one():
    """predict_single probabilities sum to approximately 1.0."""
    model = MalTwinCNN(num_classes=NUM_CLASSES)
    model.eval()
    img_array = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    class_names = [f"Family_{i}" for i in range(NUM_CLASSES)]
    device = torch.device("cpu")
    result = predict_single(model, img_array, class_names, device)
    total_prob = sum(result["probabilities"].values())
    assert abs(total_prob - 1.0) < 1e-5, f"Probabilities sum to {total_prob}, expected ~1.0"


def test_predict_single_predicted_family_in_class_names():
    """predict_single predicted_family is always a valid class name."""
    model = MalTwinCNN(num_classes=NUM_CLASSES)
    model.eval()
    img_array = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    class_names = [f"Family_{i}" for i in range(NUM_CLASSES)]
    device = torch.device("cpu")
    result = predict_single(model, img_array, class_names, device)
    assert result["predicted_family"] in class_names
