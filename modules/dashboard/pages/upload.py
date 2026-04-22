import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import streamlit as st
import plotly.graph_objects as go

import config
from modules.binary_to_image.converter import BinaryConverter
from modules.binary_to_image.utils import (
    compute_pixel_histogram,
    compute_sha256,
    validate_binary_format,
)


def render():
    """
    Binary Upload & Visualization page — implements SRS Mockup M3.
    """
    st.title("Binary Upload & Visualization")
    st.markdown(
        "Upload a PE (.exe, .dll) or ELF binary. "
        "Extensionless ELF files can be renamed to `.elf` before uploading."
    )

    uploaded_file = st.file_uploader(
        "Upload Binary File",
        type=["exe", "dll", "elf"],
    )

    if uploaded_file is None:
        st.info("Please upload a binary file to proceed.")
        return

    file_bytes = uploaded_file.read()

    # Size check
    if len(file_bytes) > config.MAX_UPLOAD_BYTES:
        st.error(
            f"Error: File too large ({len(file_bytes):,} bytes). "
            f"Cause: Maximum allowed size is {config.MAX_UPLOAD_BYTES // (1024*1024)} MB. "
            "Action: Upload a smaller binary file."
        )
        return

    # Validate binary format
    try:
        binary_format = validate_binary_format(file_bytes)
    except ValueError as e:
        st.error(str(e))
        return

    # Convert binary to image
    try:
        converter = BinaryConverter(img_size=config.IMG_SIZE)
        img_array = converter.convert(file_bytes)
    except Exception as e:
        st.error(
            f"Error: Failed to convert binary to image. "
            f"Cause: {e}. "
            "Action: Ensure the uploaded file is a valid PE or ELF binary."
        )
        return

    # Compute SHA-256
    sha256_hex = compute_sha256(file_bytes)

    # Store in session state
    st.session_state["img_array"] = img_array
    st.session_state["file_meta"] = {
        "name": uploaded_file.name,
        "size_bytes": len(file_bytes),
        "format": binary_format,
        "sha256": sha256_hex,
    }

    # Layout: two columns
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Grayscale Visualization")
        png_bytes = converter.to_png_bytes(img_array)
        st.image(png_bytes, caption=f"128×128 grayscale — {binary_format}", use_container_width=True)

    with col_right:
        st.subheader("File Metadata")
        import pandas as pd
        meta_df = pd.DataFrame(
            [
                {"Property": "File Name", "Value": uploaded_file.name},
                {"Property": "Format", "Value": binary_format},
                {"Property": "Size", "Value": f"{len(file_bytes):,} bytes"},
                {"Property": "SHA-256", "Value": sha256_hex[:32] + "..."},
            ]
        )
        st.table(meta_df.set_index("Property"))

        st.subheader("Pixel Intensity Histogram")
        hist = compute_pixel_histogram(img_array)
        fig = go.Figure(
            go.Bar(
                x=hist["bins"],
                y=hist["counts"],
                marker_color="steelblue",
            )
        )
        fig.update_layout(
            xaxis_title="Byte Value (0–255)",
            yaxis_title="Pixel Count",
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success(
        "File processed successfully. Navigate to **Malware Detection** to run analysis."
    )
