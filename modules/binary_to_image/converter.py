import math
from pathlib import Path

import cv2
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class BinaryConverter:
    """
    Converts raw PE/ELF binary bytes to 128x128 grayscale PNG.
    SRS ref: FE-2, FE-3 of Module 2
    """

    def __init__(self, img_size: int = None):
        self.img_size = img_size if img_size is not None else config.IMG_SIZE

    def convert(self, file_bytes: bytes) -> np.ndarray:
        """
        - Read bytes as uint8 array
        - Calculate width = int(math.sqrt(len(bytes)))
        - Trim array so length is divisible by width (trim tail bytes)
        - Reshape to (height, width) 2D array
        - Resize to (img_size, img_size) using cv2.INTER_LINEAR
        - Return as uint8 numpy array shape (img_size, img_size)
        """
        if not file_bytes or len(file_bytes) < 4:
            raise ValueError("Binary file is empty or too small")

        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        width = int(math.sqrt(len(arr)))
        if width == 0:
            raise ValueError("Binary file is empty or too small")
        trimmed_len = (len(arr) // width) * width
        arr = arr[:trimmed_len]
        height = trimmed_len // width
        img = arr.reshape((height, width))
        img_resized = cv2.resize(
            img,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        return img_resized.astype(np.uint8)

    def to_png_bytes(self, img_array: np.ndarray) -> bytes:
        """Encode grayscale array to PNG bytes using cv2.imencode."""
        success, buf = cv2.imencode(".png", img_array)
        if not success:
            raise RuntimeError("cv2.imencode failed to encode image")
        return buf.tobytes()

    def save(self, img_array: np.ndarray, output_path: Path) -> None:
        """Save PNG to disk using cv2.imwrite."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_array)
