"""
Tests for Module 2 — Binary-to-Image Conversion.
"""
import hashlib
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.binary_to_image.utils import (
    compute_pixel_histogram,
    compute_sha256,
    validate_binary_format,
)
from modules.binary_to_image.converter import BinaryConverter

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_pe_validation_accepts_valid_pe():
    """validate_binary_format returns 'PE' for MZ-header bytes."""
    pe_bytes = b"MZ" + b"\x00" * 512
    result = validate_binary_format(pe_bytes)
    assert result == "PE"


def test_elf_validation_accepts_valid_elf():
    """validate_binary_format returns 'ELF' for ELF-header bytes."""
    elf_bytes = b"\x7fELF" + b"\x00" * 512
    result = validate_binary_format(elf_bytes)
    assert result == "ELF"


def test_invalid_format_raises():
    """validate_binary_format raises ValueError for random/unknown bytes."""
    random_bytes = b"\x00\x01\x02\x03" + b"\xAB" * 100
    with pytest.raises(ValueError):
        validate_binary_format(random_bytes)


def test_output_shape_is_128x128():
    """BinaryConverter.convert() produces a 128x128 uint8 array."""
    pe_bytes = b"MZ" + b"\xAB" * 4096
    converter = BinaryConverter(img_size=128)
    img = converter.convert(pe_bytes)
    assert img.shape == (128, 128)
    assert img.dtype == np.uint8


def test_sha256_deterministic():
    """compute_sha256 returns the same hash for the same input."""
    data = b"MZ" + b"\xDE\xAD\xBE\xEF" * 256
    hash1 = compute_sha256(data)
    hash2 = compute_sha256(data)
    assert hash1 == hash2
    assert len(hash1) == 64  # 32 bytes = 64 hex chars
    assert hash1 == hashlib.sha256(data).hexdigest()


def test_empty_bytes_raises_value_error():
    """BinaryConverter.convert() raises ValueError for empty or tiny input."""
    converter = BinaryConverter(img_size=128)
    with pytest.raises(ValueError):
        converter.convert(b"")
    with pytest.raises(ValueError):
        converter.convert(b"\x00\x01")


def test_histogram_has_256_bins():
    """compute_pixel_histogram returns exactly 256 bins and counts."""
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    hist = compute_pixel_histogram(img)
    assert len(hist["bins"]) == 256
    assert len(hist["counts"]) == 256
    assert hist["bins"][0] == 0
    assert hist["bins"][255] == 255


def test_to_png_bytes_returns_valid_bytes():
    """BinaryConverter.to_png_bytes() returns non-empty bytes starting with PNG magic."""
    pe_bytes = b"MZ" + b"\xAB" * 4096
    converter = BinaryConverter(img_size=128)
    img = converter.convert(pe_bytes)
    png_bytes = converter.to_png_bytes(img)
    assert len(png_bytes) > 0
    # PNG magic bytes
    assert png_bytes[:4] == b"\x89PNG"
