import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import streamlit as st
import plotly.graph_objects as go

import config
from modules.detection.inference import predict_single
from modules.dashboard.db import init_db, log_detection_event


def render():
    """
    Malware Detection & Prediction page — implements SRS Mockup M5.
    """
    st.title("Malware Detection")

    # Guards
    if st.session_state.get("img_array") is None:
        st.warning("Please upload a binary file first (go to **Binary Upload** page).")
        return

    if st.session_state.get("model") is None:
        st.warning(
            "No trained model found. "
            "Train the model first using: `python scripts/train.py`"
        )
        return

    model = st.session_state["model"]
    img_array = st.session_state["img_array"]
    file_meta = st.session_state.get("file_meta", {})
    class_names = st.session_state.get("class_names") or []

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Uploaded Binary")
        from modules.binary_to_image.converter import BinaryConverter
        converter = BinaryConverter(img_size=config.IMG_SIZE)
        png_bytes = converter.to_png_bytes(img_array)
        st.image(png_bytes, caption="128×128 grayscale", use_container_width=True)

    with col_right:
        st.subheader("File Summary")
        if file_meta:
            import pandas as pd
            rows = [{"Property": k, "Value": str(v)} for k, v in file_meta.items()]
            st.table(pd.DataFrame(rows).set_index("Property"))

    if st.button("Run Detection", type="primary"):
        with st.spinner("Running inference..."):
            result = predict_single(model, img_array, class_names, config.DEVICE)
        st.session_state["detection_result"] = result

        # Log to SQLite
        try:
            init_db(config.DB_PATH)
            log_detection_event(
                db_path=config.DB_PATH,
                file_name=file_meta.get("name", "unknown"),
                sha256=file_meta.get("sha256", ""),
                file_format=file_meta.get("format", ""),
                file_size=file_meta.get("size_bytes", 0),
                predicted_family=result["predicted_family"],
                confidence=result["confidence"],
                device=str(config.DEVICE),
            )
        except Exception as e:
            st.warning(f"Could not log detection event: {e}")

    result = st.session_state.get("detection_result")
    if result is None:
        return

    st.divider()
    st.subheader("Detection Results")

    family = result["predicted_family"]
    confidence = result["confidence"]
    conf_pct = int(confidence * 100)

    # Color-coded confidence display
    if confidence >= config.CONFIDENCE_GREEN:
        st.success(f"**Predicted Family: {family}**")
    elif confidence >= config.CONFIDENCE_AMBER:
        st.warning(f"**Predicted Family: {family}** — Low confidence — verify manually")
    else:
        st.error(f"**Predicted Family: {family}** — Very low confidence — manual review required")

    # Confidence bar
    col_bar, col_pct = st.columns([4, 1])
    with col_bar:
        st.progress(conf_pct)
    with col_pct:
        st.write(f"**{conf_pct}%**")

    # Per-class probability chart
    st.subheader("Class Probability Distribution")
    probs = result.get("probabilities", {})
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_probs]
    values = [p[1] for p in sorted_probs]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="cornflowerblue",
        )
    )
    fig.update_layout(
        xaxis_title="Probability",
        yaxis_title="Malware Family",
        xaxis=dict(range=[0, 1]),
        height=max(400, len(names) * 20),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # MITRE ATT&CK mapping
    st.subheader("MITRE ATT&CK for ICS Mapping")
    mitre_path = config.MITRE_JSON_PATH
    if mitre_path.exists():
        with open(mitre_path, "r") as f:
            mitre_data = json.load(f)
        mapping = mitre_data.get(family)
        if mapping:
            tactics = mapping.get("tactics", [])
            techniques = mapping.get("techniques", [])
            st.info(f"**Tactics:** {', '.join(tactics)}")
            for tech in techniques:
                st.info(f"🔹 `{tech['id']}` — {tech['name']}")
        else:
            st.info("MITRE ATT&CK mapping not available for this family.")
    else:
        st.info("MITRE ATT&CK mapping not available for this family.")

    # XAI Heatmap stub
    st.subheader("Explainability (Grad-CAM)")
    show_gradcam = st.checkbox("Generate Grad-CAM Heatmap (requires trained model)")
    if show_gradcam:
        st.info(
            "Grad-CAM XAI will be available in the next implementation phase (Module 7)."
        )

    # Report download stub
    st.subheader("Forensic Report")
    st.download_button(
        label="Download PDF Report (Coming Soon)",
        data=b"",
        file_name="report.pdf",
        mime="application/pdf",
        disabled=True,
    )
