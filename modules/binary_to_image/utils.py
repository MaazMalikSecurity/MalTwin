import hashlib
import numpy as np


def validate_binary_format(file_bytes: bytes) -> str:
    """
    Returns 'PE', 'ELF', or raises ValueError.
    PE magic: first 2 bytes == b'MZ'
    ELF magic: first 4 bytes == b'\\x7fELF'
    SRS ref: UC-01 Alternate Flow A1, FR-B1
    """
    if len(file_bytes) < 4:
        raise ValueError("Binary file is empty or too small")
    if file_bytes[:2] == b"MZ":
        return "PE"
    if file_bytes[:4] == b"\x7fELF":
        return "ELF"
    raise ValueError(
        "Error: Unrecognised binary format. "
        "Cause: File does not begin with PE (MZ) or ELF magic bytes. "
        "Action: Ensure you upload a valid .exe, .dll, or ELF binary."
    )


def compute_sha256(file_bytes: bytes) -> str:
    """
    Returns lowercase hex SHA-256 digest.
    Uses hashlib.sha256 only — no external services.
    SRS ref: SRS FR3.3, SEC-4, CON-9
    """
    return hashlib.sha256(file_bytes).hexdigest()


def compute_pixel_histogram(img_array: np.ndarray) -> dict:
    """
    Returns dict with keys 'bins' (list 0-255) and 'counts' (list of int).
    256 bins exactly, one per byte value.
    SRS ref: SRS FR3.4
    """
    flat = img_array.flatten().astype(np.uint8)
    counts, _ = np.histogram(flat, bins=256, range=(0, 256))
    return {
        "bins": list(range(256)),
        "counts": counts.tolist(),
    }
