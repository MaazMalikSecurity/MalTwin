import streamlit as st


def render():
    """
    Home / Dashboard overview page — M1 Digital Twin status (stubbed).
    """
    st.title("MalTwin — Malware Intelligence Platform")
    st.markdown(
        """
        Welcome to **MalTwin**, an intelligent malware detection platform using
        binary-to-image conversion and CNN-based classification.
        """
    )

    st.subheader("Module Status")
    col1, col2 = st.columns(2)

    with col1:
        st.info("✅ **M2** — Binary-to-Image Conversion: Active")
        st.info("✅ **M3** — Dataset Collection & Preprocessing: Active")
        st.info("✅ **M4** — Data Enhancement & Balancing: Active")
        st.info("✅ **M5** — Intelligent Malware Detection (CNN): Active")

    with col2:
        st.info("🟡 **M6** — Dashboard & Visualization: Partial")
        st.warning("❌ **M1** — Digital Twin Simulation: Deferred (requires Docker + Mininet)")
        st.warning("❌ **M7** — Explainable AI (Grad-CAM): Deferred")
        st.warning("❌ **M8** — Automated Threat Reporting: Deferred")

    st.divider()
    st.subheader("Digital Twin Monitor (Coming Soon)")
    st.info(
        "The Digital Twin simulation module (M1) requires Docker + Mininet infrastructure. "
        "This tab will show live network traffic analysis from the simulated OT/ICS environment "
        "once M1 is implemented."
    )

    # Show recent detection events if DB exists
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../..", ""))
    import config
    from modules.dashboard.db import get_recent_events, init_db

    if config.DB_PATH.exists():
        st.divider()
        st.subheader("Recent Detection Events")
        events = get_recent_events(config.DB_PATH, limit=5)
        if events:
            import pandas as pd
            st.dataframe(pd.DataFrame(events))
        else:
            st.info("No detection events logged yet.")
