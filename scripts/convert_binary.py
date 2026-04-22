"""
CLI single-file binary-to-image conversion utility.
Usage: python scripts/convert_binary.py --input path/to/file.exe --output path/to/out.png
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Convert binary file to grayscale PNG")
    parser.add_argument("--input", required=True, help="Path to input binary file")
    parser.add_argument("--output", required=True, help="Path to output PNG file")
    args = parser.parse_args()

    from pathlib import Path
    from modules.binary_to_image.utils import validate_binary_format, compute_sha256
    from modules.binary_to_image.converter import BinaryConverter
    import config

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    file_bytes = input_path.read_bytes()

    try:
        fmt = validate_binary_format(file_bytes)
        print(f"Detected format: {fmt}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    sha256_hex = compute_sha256(file_bytes)
    print(f"SHA-256: {sha256_hex}")

    converter = BinaryConverter(img_size=config.IMG_SIZE)
    img_array = converter.convert(file_bytes)
    converter.save(img_array, output_path)
    print(f"Saved 128x128 grayscale image to {output_path}")


if __name__ == "__main__":
    main()
