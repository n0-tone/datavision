from __future__ import annotations

import streamlit as st

from src.components.sidebar import render_datavision_sidebar

from .auth import require_login
from .data_ops import apply_sidebar_filters, load_csv, split_columns
from .theme import apply_theme
from .visuals import (
    render_advanced_stats_tab,
    render_clustering_tab,
    render_feature_importance_tab,
    render_hero,
    render_insights_tab,
    render_overview_tab,
    render_pca_tab,
    render_quality_tab,
    render_relationships_tab,
    render_supervised_models_tab,
    render_time_series_tab,
)


def run_app() -> None:
    st.set_page_config(
        page_title="no-tone | DataVision",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_theme()
    require_login()

    logout_clicked, uploaded_file = render_datavision_sidebar()

    if logout_clicked:
        st.session_state.authenticated = False
        st.rerun()

    if uploaded_file is None:
        st.markdown("### Upload a CSV to begin")
        st.info("Use the file uploader in the sidebar. The dashboard updates automatically.")
        return

    try:
        df = load_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to parse CSV: {exc}")
        return

    if df.empty:
        st.warning("The uploaded file has no rows.")
        return

    filtered_df = apply_sidebar_filters(df)
    if filtered_df.empty:
        st.warning("No rows left after filtering. Adjust sidebar filters.")
        return

    numeric, categorical = split_columns(filtered_df)

    render_hero(filtered_df)

    (
        tab_overview,
        tab_stats,
        tab_quality,
        tab_relationships,
        tab_clustering,
        tab_pca,
        tab_importance,
        tab_supervised,
        tab_time,
        tab_insights,
    ) = st.tabs(
        [
            "Overview",
            "Advanced Stats",
            "Data Quality",
            "Relationships",
            "Clustering",
            "PCA",
            "Feature Importance",
            "Linear/Logistic/Tree/RF",
            "Time Series",
            "Auto Insights",
        ]
    )

    with tab_overview:
        render_overview_tab(filtered_df, numeric, categorical)
    with tab_stats:
        render_advanced_stats_tab(filtered_df, numeric)
    with tab_quality:
        render_quality_tab(filtered_df)
    with tab_relationships:
        render_relationships_tab(filtered_df, numeric)
    with tab_clustering:
        render_clustering_tab(filtered_df, numeric)
    with tab_pca:
        render_pca_tab(filtered_df, numeric)
    with tab_importance:
        render_feature_importance_tab(filtered_df)
    with tab_supervised:
        render_supervised_models_tab(filtered_df)
    with tab_time:
        render_time_series_tab(filtered_df, numeric)
    with tab_insights:
        render_insights_tab(filtered_df, numeric)
