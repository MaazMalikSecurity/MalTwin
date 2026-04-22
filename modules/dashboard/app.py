"""
Streamlit entry point for MalTwin dashboard.
Run with: streamlit run modules/dashboard/app.py --server.port 8501
"""
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

import config
from modules.dashboard.db import init_db


def load_class_names() -> list:
    class_names_path = config.PROCESSED_DIR / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path, "r") as f:
            return json.load(f)
    return None


def load_model_if_available(class_names: list):
    if class_names is None:
        return None
    if not config.BEST_MODEL_PATH.exists():
        return None
    try:
        from modules.detection.inference import load_model
        model = load_model(config.BEST_MODEL_PATH, len(class_names), config.DEVICE)
        return model
    except Exception:
        return None


def main():
    st.set_page_config(
        page_title="MalTwin",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialise session state keys once
    if "model" not in st.session_state:
        class_names = load_class_names()
        st.session_state["class_names"] = class_names
        st.session_state["model"] = load_model_if_available(class_names)

    if "img_array" not in st.session_state:
        st.session_state["img_array"] = None

    if "file_meta" not in st.session_state:
        st.session_state["file_meta"] = None

    if "detection_result" not in st.session_state:
        st.session_state["detection_result"] = None

    # Ensure DB and log dir exist
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        init_db(config.DB_PATH)
    except Exception:
        pass

    # Sidebar navigation
    st.sidebar.title("MalTwin")
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox(
        "Navigate",
        ["Dashboard", "Binary Upload", "Malware Detection", "Digital Twin (Coming Soon)"],
    )

    # Model/class_names status in sidebar
    if st.session_state.get("model") is not None:
        st.sidebar.success("Model loaded ✅")
    else:
        st.sidebar.warning("No model loaded ⚠️")

    if st.session_state.get("class_names") is not None:
        st.sidebar.info(f"{len(st.session_state['class_names'])} malware families")

    st.sidebar.markdown("---")
    st.sidebar.caption("MalTwin v0.1 — 30% Implementation")

    # Page routing
    if page == "Dashboard":
        from modules.dashboard.pages.home import render
        render()
    elif page == "Binary Upload":
        from modules.dashboard.pages.upload import render
        render()
    elif page == "Malware Detection":
        from modules.dashboard.pages.detection import render
        render()
    elif page == "Digital Twin (Coming Soon)":
        st.title("Digital Twin Simulation (Coming Soon)")
        st.info(
            "This module requires Docker + Mininet infrastructure (M1). "
            "It will provide live OT/ICS network traffic simulation and monitoring."
        )


if __name__ == "__main__":
    main()
