from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .data_ops import build_auto_insights, build_outlier_table, detect_datetime_candidates


def card_metric(container, label: str, value: str) -> None:
    container.markdown(
        f"""
        <div class="metric-shell">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin:0;">DataVision Analytics Studio</h1>
            <p style="margin:8px 0 0;color:#96a9c3;">
                Minimal dark workspace for exploratory analytics, data quality checks, and machine learning visuals.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    numeric_count = len(df.select_dtypes(include="number").columns)

    c1, c2, c3, c4 = st.columns(4)
    card_metric(c1, "Rows", f"{len(df):,}")
    card_metric(c2, "Columns", str(df.shape[1]))
    card_metric(c3, "Numeric Features", str(numeric_count))
    card_metric(c4, "Missing / Duplicates", f"{missing:,} / {duplicates:,}")


def render_overview_tab(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> None:
    st.subheader("Data Overview")
    left, right = st.columns([1, 1.15])

    info = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(v) for v in df.dtypes],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "missing_pct": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )

    with left:
        st.markdown("##### Column Profile")
        st.dataframe(info, width="stretch", hide_index=True)

    with right:
        st.markdown("##### Data Preview")
        preview_size = st.slider("Preview rows", 10, 200, 30, key="overview_preview")
        st.dataframe(df.head(preview_size), width="stretch")

    viz_left, viz_right = st.columns(2)
    with viz_left:
        st.markdown("##### Quick Histogram")
        if numeric:
            hist_col = st.selectbox("Numeric feature", numeric, key="overview_hist_col")
            fig = px.histogram(
                df,
                x=hist_col,
                nbins=40,
                color_discrete_sequence=["#2de2c4"],
                title=f"Distribution of {hist_col}",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No numeric columns available.")

    with viz_right:
        st.markdown("##### Top Categories")
        if categorical:
            cat_col = st.selectbox("Categorical feature", categorical, key="overview_cat_col")
            top_n = st.slider("Top categories", 5, 30, 12, key="overview_top_n")
            vc = (
                df[cat_col]
                .fillna("<missing>")
                .astype(str)
                .value_counts()
                .head(top_n)
                .sort_values(ascending=True)
            )
            fig = px.bar(
                vc,
                x=vc.values,
                y=vc.index,
                orientation="h",
                color=vc.values,
                color_continuous_scale=["#182335", "#2de2c4"],
                title=f"Top {top_n} categories in {cat_col}",
            )
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No categorical columns available.")


def render_advanced_stats_tab(df: pd.DataFrame, numeric: list[str]) -> None:
    st.subheader("Advanced Statistics")
    if not numeric:
        st.warning("No numeric columns available for statistics.")
        return

    selected = st.multiselect(
        "Features",
        options=numeric,
        default=numeric[: min(8, len(numeric))],
        key="stats_features",
    )
    if not selected:
        st.info("Select at least one numeric feature.")
        return

    stats = df[selected].agg(["count", "mean", "median", "std", "min", "max", "skew", "kurt"]).T
    stats["iqr"] = (df[selected].quantile(0.75) - df[selected].quantile(0.25)).values
    st.dataframe(stats.round(4), width="stretch")

    outlier_df = build_outlier_table(df, selected)
    st.markdown("##### Outlier Detection (IQR Rule)")
    st.dataframe(outlier_df, width="stretch", hide_index=True)

    b1, b2 = st.columns(2)
    with b1:
        box_col = st.selectbox("Boxplot feature", selected, key="stats_box_col")
        category_options = ["None"] + [c for c in df.columns if c not in selected]
        split = st.selectbox("Split by category", category_options, key="stats_split")

        if split == "None":
            fig = px.box(
                df,
                y=box_col,
                points="outliers",
                color_discrete_sequence=["#f5ae57"],
                title=f"Boxplot: {box_col}",
            )
        else:
            fig = px.box(df, x=split, y=box_col, points="outliers", title=f"Boxplot: {box_col} by {split}")

        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, width="stretch")

    with b2:
        stat_col = st.selectbox("Distribution feature", selected, key="stats_dist_col")
        fig = px.histogram(
            df,
            x=stat_col,
            marginal="box",
            color_discrete_sequence=["#2de2c4"],
            title=f"Distribution + Outlier Context: {stat_col}",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, width="stretch")


def render_quality_tab(df: pd.DataFrame) -> None:
    st.subheader("Data Quality")

    missing = (
        df.isna()
        .sum()
        .rename("missing")
        .to_frame()
        .assign(missing_pct=lambda x: (x["missing"] / len(df) * 100).round(2))
        .sort_values("missing", ascending=False)
    )

    strict_complete_rows = int((~df.isna().any(axis=1)).sum())
    quality_cols = [
        c
        for c in df.columns
        if df[c].notna().any() and not str(c).strip().lower().startswith("unnamed:")
    ]
    if not quality_cols:
        quality_cols = [c for c in df.columns if df[c].notna().any()]
    if not quality_cols:
        quality_cols = df.columns.tolist()

    complete_rows = int((~df[quality_cols].isna().any(axis=1)).sum())

    q1, q2, q3 = st.columns(3)
    q1.metric("Total Missing", f"{int(missing['missing'].sum()):,}")
    q2.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
    q3.metric("Complete Rows", f"{complete_rows:,}")

    if strict_complete_rows != complete_rows:
        st.caption(
            f"Strict all-column complete rows: {strict_complete_rows:,}. "
            "Displayed complete rows ignore fully empty helper columns."
        )

    left, right = st.columns([1, 1.1])
    with left:
        st.markdown("##### Missing Values by Feature")
        table = missing.reset_index().rename(columns={"index": "column"})
        st.dataframe(table, width="stretch")

    with right:
        st.markdown("##### Data Type Distribution")
        dtype_counts = df.dtypes.astype(str).value_counts().rename_axis("dtype").reset_index(name="count")
        fig_types = px.pie(
            dtype_counts,
            names="dtype",
            values="count",
            hole=0.48,
            title="Column Data Types",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_types.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_types, width="stretch")

    st.markdown("##### Missing Value Heatmap")
    if df.shape[1] > 1:
        max_rows = min(500, len(df))
        max_cols = min(30, df.shape[1])
        sample = df.iloc[:max_rows, :max_cols].isna().astype(int).T
        fig_missing = px.imshow(
            sample,
            aspect="auto",
            color_continuous_scale=[[0, "#0f1723"], [1, "#f16d86"]],
            labels={"x": "Row Index", "y": "Feature", "color": "Missing"},
            title=f"First {max_rows} rows x {max_cols} columns",
        )
        fig_missing.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=430)
        st.plotly_chart(fig_missing, width="stretch")


def render_relationships_tab(df: pd.DataFrame, numeric: list[str]) -> None:
    st.subheader("Relationships and Correlations")
    if len(numeric) < 2:
        st.warning("At least two numeric columns are required.")
        return

    method = st.radio("Correlation method", ["pearson", "spearman", "kendall"], horizontal=True)
    features = st.multiselect(
        "Numeric features for correlation",
        options=numeric,
        default=numeric[: min(10, len(numeric))],
        key="rel_features",
    )
    if len(features) < 2:
        st.info("Select at least two numeric features.")
        return

    corr = df[features].corr(method=method)
    fig_heatmap = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title=f"Correlation Heatmap ({method.title()})",
    )
    fig_heatmap.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=560)
    st.plotly_chart(fig_heatmap, width="stretch")

    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index(name="correlation")
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    pairs["abs_correlation"] = pairs["correlation"].abs()
    threshold = st.slider("Minimum absolute correlation", 0.0, 1.0, 0.5, 0.05)
    top_pairs = pairs[pairs["abs_correlation"] >= threshold].sort_values("abs_correlation", ascending=False)

    st.markdown("##### Top Correlated Feature Pairs")
    if top_pairs.empty:
        st.info("No pairs above the selected threshold.")
    else:
        st.dataframe(top_pairs.head(25), width="stretch", hide_index=True)

    st.markdown("##### Scatter Matrix")
    matrix_features = st.multiselect(
        "Scatter matrix features",
        options=features,
        default=features[: min(4, len(features))],
        key="rel_matrix_features",
    )
    color_options = ["None"] + [c for c in df.columns if c not in matrix_features]
    color_col = st.selectbox("Color by", options=color_options, key="rel_matrix_color")
    sample_size = st.slider("Scatter matrix sample size", 300, 4000, 1500, 100)

    if len(matrix_features) >= 2:
        sample_df = df[matrix_features + ([color_col] if color_col != "None" else [])].dropna()
        if not sample_df.empty:
            if len(sample_df) > sample_size:
                sample_df = sample_df.sample(sample_size, random_state=42)

            if color_col == "None":
                fig_matrix = px.scatter_matrix(sample_df, dimensions=matrix_features, title="Scatter Matrix")
            else:
                fig_matrix = px.scatter_matrix(
                    sample_df,
                    dimensions=matrix_features,
                    color=color_col,
                    title="Scatter Matrix",
                )

            fig_matrix.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.6))
            fig_matrix.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=760)
            st.plotly_chart(fig_matrix, width="stretch")


def render_clustering_tab(df: pd.DataFrame, numeric: list[str]) -> None:
    st.subheader("Clustering")
    if len(numeric) < 2:
        st.warning("At least two numeric columns are needed for clustering.")
        return

    features = st.multiselect(
        "Clustering features",
        options=numeric,
        default=numeric[: min(5, len(numeric))],
        key="cluster_features",
    )
    if len(features) < 2:
        st.info("Select at least two features.")
        return

    missing_strategy = st.selectbox(
        "Missing value handling",
        options=["Median imputation", "Drop rows with missing values"],
        key="cluster_missing_strategy",
    )

    raw_matrix = df[features].apply(pd.to_numeric, errors="coerce")
    usable_features = [col for col in raw_matrix.columns if raw_matrix[col].notna().any()]
    if len(usable_features) < 2:
        st.warning("Selected features do not contain enough numeric data.")
        return
    if len(usable_features) != len(features):
        skipped = sorted(set(features) - set(usable_features))
        st.caption(f"Ignored empty/all-missing features: {', '.join(skipped)}")

    raw_matrix = raw_matrix[usable_features]
    if missing_strategy == "Drop rows with missing values":
        cluster_df = raw_matrix.dropna()
    else:
        medians = raw_matrix.median()
        cluster_df = raw_matrix.fillna(medians).fillna(0.0)

    if len(cluster_df) < 6:
        st.warning("Not enough valid rows after applying missing-value handling.")
        return

    max_k = min(12, len(cluster_df) - 1)
    k = st.slider("Number of clusters (k)", 2, max_k, min(4, max_k), key="cluster_k")
    standardize = st.checkbox("Standardize features", value=True, key="cluster_standardize")

    model_input = cluster_df.copy()
    if standardize:
        model_input = pd.DataFrame(
            StandardScaler().fit_transform(model_input),
            columns=cluster_df.columns,
            index=cluster_df.index,
        )

    if st.checkbox("Show elbow curve", value=True, key="cluster_elbow"):
        max_elbow_k = min(10, max_k)
        ks = list(range(2, max_elbow_k + 1))
        inertias = []
        for n in ks:
            km = KMeans(n_clusters=n, n_init="auto", random_state=42)
            km.fit(model_input)
            inertias.append(km.inertia_)

        elbow_fig = px.line(
            x=ks,
            y=inertias,
            markers=True,
            labels={"x": "k", "y": "Inertia"},
            title="Elbow Curve",
        )
        elbow_fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=330)
        st.plotly_chart(elbow_fig, width="stretch")

    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(model_input)
    silhouette = silhouette_score(model_input, labels)

    m1, m2 = st.columns(2)
    m1.metric("Inertia", f"{model.inertia_:,.2f}")
    m2.metric("Silhouette Score", f"{silhouette:.3f}")

    projection = PCA(n_components=2, random_state=42)
    reduced = projection.fit_transform(model_input)
    projected = pd.DataFrame(reduced, columns=["pc1", "pc2"], index=model_input.index)
    projected["cluster"] = labels.astype(str)

    exp_var = projection.explained_variance_ratio_.sum() * 100
    st.caption(f"PCA projection variance explained: {exp_var:.1f}%")

    fig = px.scatter(
        projected,
        x="pc1",
        y="pc2",
        color="cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Cluster Projection",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig, width="stretch")

    profile = cluster_df.assign(cluster=labels).groupby("cluster")[usable_features].mean().round(3)
    st.markdown("##### Cluster Profiles")
    st.dataframe(profile, width="stretch")


def render_pca_tab(df: pd.DataFrame, numeric: list[str]) -> None:
    st.subheader("Principal Component Analysis")
    if len(numeric) < 2:
        st.warning("At least two numeric columns are needed for PCA.")
        return

    features = st.multiselect(
        "PCA features",
        options=numeric,
        default=numeric[: min(8, len(numeric))],
        key="pca_features",
    )
    if len(features) < 2:
        st.info("Select at least two numeric features.")
        return

    missing_strategy = st.selectbox(
        "Missing value handling",
        options=["Median imputation", "Drop rows with missing values"],
        key="pca_missing_strategy",
    )

    raw_matrix = df[features].apply(pd.to_numeric, errors="coerce")
    usable_features = [col for col in raw_matrix.columns if raw_matrix[col].notna().any()]
    if len(usable_features) < 2:
        st.warning("Selected features do not contain enough numeric data.")
        return
    if len(usable_features) != len(features):
        skipped = sorted(set(features) - set(usable_features))
        st.caption(f"Ignored empty/all-missing features: {', '.join(skipped)}")

    raw_matrix = raw_matrix[usable_features]
    if missing_strategy == "Drop rows with missing values":
        clean = raw_matrix.dropna()
    else:
        medians = raw_matrix.median()
        clean = raw_matrix.fillna(medians).fillna(0.0)

    if len(clean) < 5:
        st.warning("Not enough valid rows after applying missing-value handling.")
        return

    do_scale = st.checkbox("Standardize features before PCA", value=True, key="pca_scale")
    matrix = clean.copy()
    if do_scale:
        matrix = pd.DataFrame(StandardScaler().fit_transform(matrix), columns=usable_features, index=clean.index)

    max_comp = min(10, len(features), len(matrix))
    comp_n = st.slider("Number of components", 2, max_comp, min(4, max_comp), key="pca_components")

    pca = PCA(n_components=comp_n, random_state=42)
    transformed = pca.fit_transform(matrix)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    var_df = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(comp_n)],
            "explained": explained,
            "cumulative": cumulative,
        }
    )

    left, right = st.columns(2)
    with left:
        fig_bar = px.bar(
            var_df,
            x="component",
            y="explained",
            title="Explained Variance by Component",
            color_discrete_sequence=["#2de2c4"],
        )
        fig_bar.update_layout(yaxis_tickformat=".1%", margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_bar, width="stretch")

    with right:
        fig_line = px.line(
            var_df,
            x="component",
            y="cumulative",
            markers=True,
            title="Cumulative Explained Variance",
        )
        fig_line.update_layout(yaxis_tickformat=".1%", margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_line, width="stretch")

    plot_df = pd.DataFrame(transformed[:, :2], columns=["PC1", "PC2"], index=matrix.index)
    color_options = ["None"] + df.columns.tolist()
    color_col = st.selectbox("Color PCA scatter by", color_options, key="pca_color")

    if color_col == "None":
        fig_scatter = px.scatter(plot_df, x="PC1", y="PC2", title="PCA Projection", opacity=0.8)
    else:
        color_series = df.loc[plot_df.index, color_col]
        fig_scatter = px.scatter(
            plot_df.assign(color=color_series.values),
            x="PC1",
            y="PC2",
            color="color",
            title="PCA Projection",
            opacity=0.8,
        )

    fig_scatter.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig_scatter, width="stretch")


def render_feature_importance_tab(df: pd.DataFrame) -> None:
    st.subheader("Feature Importance")
    st.caption("Model-based ranking using Random Forest.")

    target = st.selectbox("Target column", options=df.columns.tolist(), key="fi_target")
    candidate_features = [c for c in df.columns if c != target]
    if not candidate_features:
        st.warning("No feature columns available.")
        return

    features = st.multiselect(
        "Feature columns",
        options=candidate_features,
        default=candidate_features[: min(8, len(candidate_features))],
        key="fi_features",
    )
    if not features:
        st.info("Select at least one feature column.")
        return

    mode = st.selectbox("Task type", ["Auto", "Classification", "Regression"], key="fi_mode")
    model_rows = st.slider("Max rows for training", 500, 30000, 8000, 500, key="fi_rows")

    missing_strategy = st.selectbox(
        "Missing value handling",
        options=["Median/Mode imputation", "Drop rows with missing values"],
        key="fi_missing_strategy",
    )

    x_data = df[features].copy()
    y_raw = df[target].copy()

    if missing_strategy == "Drop rows with missing values":
        valid_mask = x_data.notna().all(axis=1) & y_raw.notna()
        x_data = x_data.loc[valid_mask]
        y_raw = y_raw.loc[valid_mask]
    else:
        for col in x_data.columns:
            if pd.api.types.is_numeric_dtype(x_data[col]):
                numeric_col = pd.to_numeric(x_data[col], errors="coerce")
                if numeric_col.notna().any():
                    median_value = float(numeric_col.median())
                else:
                    median_value = 0.0
                x_data[col] = numeric_col.fillna(median_value)
            else:
                mode_values = x_data[col].mode(dropna=True)
                fill_value = mode_values.iloc[0] if not mode_values.empty else "<missing>"
                x_data[col] = x_data[col].astype(str).replace("nan", np.nan).fillna(fill_value)

    if len(x_data) < 30:
        st.warning("Not enough valid rows for feature importance modeling.")
        return

    if len(x_data) > model_rows:
        sampled_idx = x_data.sample(model_rows, random_state=42).index
        x_data = x_data.loc[sampled_idx]
        y_raw = y_raw.loc[sampled_idx]

    x_train = pd.get_dummies(x_data, drop_first=True)
    if x_train.empty:
        st.warning("Feature encoding produced an empty matrix.")
        return

    resolved_mode = mode
    if mode == "Auto":
        if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > 20:
            resolved_mode = "Regression"
        else:
            resolved_mode = "Classification"

    if resolved_mode == "Regression":
        y_num = pd.to_numeric(y_raw, errors="coerce")
        valid_mask = y_num.notna()
        x_fit = x_train.loc[valid_mask]
        y_fit = y_num.loc[valid_mask]
        if len(y_fit) < 30:
            st.warning("Target is not suitable for regression after numeric conversion.")
            return
        model = RandomForestRegressor(n_estimators=240, random_state=42, n_jobs=-1)
        model.fit(x_fit, y_fit)
    else:
        y_fit, classes = pd.factorize(y_raw.fillna("<missing>").astype(str))
        if len(np.unique(y_fit)) < 2:
            st.warning("Classification target needs at least two classes.")
            return
        model = RandomForestClassifier(n_estimators=260, random_state=42, n_jobs=-1)
        model.fit(x_train, y_fit)
        st.caption(f"Detected classes: {len(classes)}")
        x_fit = x_train

    imp = (
        pd.DataFrame({"feature": x_fit.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(25)
    )
    fig = px.bar(
        imp.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#182335", "#2de2c4"],
        title=f"Top Feature Importance ({resolved_mode})",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(imp, width="stretch", hide_index=True)


def render_supervised_models_tab(df: pd.DataFrame) -> None:
    st.subheader("Supervised Models")
    st.caption("Train and evaluate linear/logistic regression, decision tree, and random forest.")

    target = st.selectbox("Target column", options=df.columns.tolist(), key="sup_target")
    candidate_features = [
        c
        for c in df.columns
        if c != target and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if not candidate_features:
        st.warning("No usable feature columns available.")
        return

    default_features = [
        c
        for c in candidate_features
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True) <= 80
    ]
    if not default_features:
        default_features = candidate_features

    features = st.multiselect(
        "Feature columns",
        options=candidate_features,
        default=default_features[: min(8, len(default_features))],
        key="sup_features",
    )
    if not features:
        st.info("Select at least one feature column.")
        return

    mode = st.selectbox("Task type", ["Auto", "Classification", "Regression"], key="sup_mode")
    missing_strategy = st.selectbox(
        "Missing value handling",
        ["Median/Mode imputation", "Drop rows with missing values"],
        key="sup_missing_strategy",
    )
    test_size = st.slider("Test set (%)", 10, 40, 20, key="sup_test_size")
    max_rows = st.slider("Max rows for training", 200, 20000, 4000, 200, key="sup_max_rows")
    max_category_levels = st.slider(
        "Max category levels per categorical feature",
        10,
        200,
        60,
        5,
        key="sup_max_category_levels",
        help="Rare categories are grouped into <other> to keep training responsive.",
    )
    max_encoded_features = st.slider(
        "Max encoded feature columns",
        100,
        3000,
        1200,
        50,
        key="sup_max_encoded_features",
        help="Hard cap after one-hot encoding to avoid memory spikes.",
    )

    resolved_mode = mode
    if mode == "Auto":
        target_preview = pd.to_numeric(df[target], errors="coerce")
        is_regression_like = (
            pd.api.types.is_numeric_dtype(df[target])
            and target_preview.notna().mean() >= 0.9
            and target_preview.nunique(dropna=True) > 14
        )
        resolved_mode = "Regression" if is_regression_like else "Classification"

    if resolved_mode == "Regression":
        model_name = st.selectbox(
            "Model",
            ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"],
            key="sup_model_reg",
        )
    else:
        model_name = st.selectbox(
            "Model",
            ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier"],
            key="sup_model_clf",
        )

    model_params: dict[str, int] = {}

    if resolved_mode == "Regression":
        if model_name == "Linear Regression":
            pass
        elif model_name == "Decision Tree Regressor":
            max_depth = st.slider("Tree max depth", 2, 40, 10, key="sup_reg_tree_depth")
            model_params["max_depth"] = max_depth
        else:
            n_estimators = st.slider("Forest trees", 50, 400, 160, 10, key="sup_reg_rf_estimators")
            max_depth = st.slider("Forest max depth", 2, 40, 12, key="sup_reg_rf_depth")
            model_params["n_estimators"] = n_estimators
            model_params["max_depth"] = max_depth
    else:
        if model_name == "Logistic Regression":
            pass
        elif model_name == "Decision Tree Classifier":
            max_depth = st.slider("Tree max depth", 2, 40, 10, key="sup_clf_tree_depth")
            model_params["max_depth"] = max_depth
        else:
            n_estimators = st.slider("Forest trees", 50, 400, 160, 10, key="sup_clf_rf_estimators")
            max_depth = st.slider("Forest max depth", 2, 40, 12, key="sup_clf_rf_depth")
            model_params["n_estimators"] = n_estimators
            model_params["max_depth"] = max_depth

    config_signature = (
        target,
        tuple(features),
        mode,
        missing_strategy,
        test_size,
        max_rows,
        max_category_levels,
        max_encoded_features,
        model_name,
        tuple(sorted(model_params.items())),
        len(df),
        tuple(df.columns),
    )

    artifact_key = "sup_model_artifact"
    signature_key = "sup_model_signature"
    status = st.empty()
    if st.button("Train / Refresh Model", key="sup_train_button"):
        with st.spinner("Training model..."):
            x_raw = df[features].copy()
            y_raw = df[target].copy()

            if missing_strategy == "Drop rows with missing values":
                valid_mask = x_raw.notna().all(axis=1) & y_raw.notna()
                x_raw = x_raw.loc[valid_mask]
                y_raw = y_raw.loc[valid_mask]

            feature_fill_values: dict[str, float | str] = {}
            category_maps: dict[str, list[str]] = {}
            numeric_features: list[str] = []
            categorical_features: list[str] = []

            for col in x_raw.columns:
                if pd.api.types.is_numeric_dtype(x_raw[col]):
                    numeric_col = pd.to_numeric(x_raw[col], errors="coerce")
                    if numeric_col.notna().any():
                        fill_value = float(numeric_col.median())
                    else:
                        fill_value = 0.0
                    feature_fill_values[col] = fill_value
                    if missing_strategy != "Drop rows with missing values":
                        x_raw[col] = numeric_col.fillna(fill_value)
                    else:
                        x_raw[col] = numeric_col
                    numeric_features.append(col)
                else:
                    text_values = x_raw[col].astype(str).replace("nan", np.nan)
                    mode_values = text_values.mode(dropna=True)
                    fill_value = mode_values.iloc[0] if not mode_values.empty else "<missing>"
                    feature_fill_values[col] = str(fill_value)
                    filled = text_values.fillna(str(fill_value))
                    top_values = filled.value_counts().head(max_category_levels).index.tolist()
                    if not top_values:
                        top_values = [str(fill_value)]
                    x_raw[col] = filled.where(filled.isin(top_values), "<other>")
                    category_maps[col] = top_values
                    categorical_features.append(col)

            resolved_mode = mode
            if mode == "Auto":
                numeric_target = pd.to_numeric(y_raw, errors="coerce")
                is_regression_like = (
                    pd.api.types.is_numeric_dtype(y_raw)
                    and numeric_target.notna().mean() >= 0.9
                    and numeric_target.nunique(dropna=True) > 14
                )
                resolved_mode = "Regression" if is_regression_like else "Classification"

            if resolved_mode == "Regression":
                y_data = pd.to_numeric(y_raw, errors="coerce")
                valid_mask = y_data.notna()
                x_data = x_raw.loc[valid_mask]
                y_data = y_data.loc[valid_mask]
                class_labels: list[str] = []
            else:
                y_labels = y_raw.fillna("<missing>").astype(str)
                y_codes, classes = pd.factorize(y_labels, sort=True)
                x_data = x_raw
                y_data = pd.Series(y_codes, index=y_labels.index)
                class_labels = [str(item) for item in classes]

            if len(x_data) < 40:
                st.warning("Not enough valid rows for supervised training.")
                return

            if len(x_data) > max_rows:
                sampled_idx = x_data.sample(max_rows, random_state=42).index
                x_data = x_data.loc[sampled_idx]
                y_data = y_data.loc[sampled_idx]

            x_encoded = pd.get_dummies(x_data, drop_first=False)
            if x_encoded.empty:
                st.warning("Feature encoding produced an empty matrix.")
                return

            dropped_encoded = 0
            if x_encoded.shape[1] > max_encoded_features:
                keep_numeric = [c for c in x_encoded.columns if c in numeric_features]
                categorical_encoded = [c for c in x_encoded.columns if c not in keep_numeric]
                remaining_budget = max(max_encoded_features - len(keep_numeric), 0)
                if remaining_budget == 0:
                    x_encoded = x_encoded[keep_numeric]
                    dropped_encoded = len(categorical_encoded)
                else:
                    freq_rank = x_encoded[categorical_encoded].sum(axis=0).sort_values(ascending=False)
                    keep_categorical = freq_rank.head(remaining_budget).index.tolist()
                    x_encoded = x_encoded[keep_numeric + keep_categorical]
                    dropped_encoded = len(categorical_encoded) - len(keep_categorical)

            test_fraction = test_size / 100.0
            stratify = None
            if resolved_mode == "Classification":
                class_counts = y_data.value_counts()
                if len(class_counts) < 2:
                    st.warning("Classification target needs at least two classes.")
                    return
                if len(class_counts) > 120:
                    st.warning(
                        "Target has too many classes for responsive training. "
                        "Try regression mode or choose a lower-cardinality target."
                    )
                    return
                if class_counts.min() >= 2:
                    stratify = y_data

            try:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_encoded,
                    y_data,
                    test_size=test_fraction,
                    random_state=42,
                    stratify=stratify,
                )
            except ValueError as exc:
                st.warning(f"Unable to split data for training: {exc}")
                return

            if resolved_mode == "Regression":
                if model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "Decision Tree Regressor":
                    model = DecisionTreeRegressor(max_depth=model_params["max_depth"], random_state=42)
                else:
                    model = RandomForestRegressor(
                        n_estimators=model_params["n_estimators"],
                        max_depth=model_params["max_depth"],
                        random_state=42,
                        n_jobs=-1,
                    )
            else:
                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1200, solver="saga", n_jobs=-1)
                elif model_name == "Decision Tree Classifier":
                    model = DecisionTreeClassifier(max_depth=model_params["max_depth"], random_state=42)
                else:
                    model = RandomForestClassifier(
                        n_estimators=model_params["n_estimators"],
                        max_depth=model_params["max_depth"],
                        random_state=42,
                        n_jobs=-1,
                    )

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            if resolved_mode == "Regression":
                metrics = {
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "r2": r2_score(y_test, y_pred),
                }
            else:
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                }

            if hasattr(model, "feature_importances_"):
                importance = pd.DataFrame(
                    {"feature": x_encoded.columns, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)
            elif hasattr(model, "coef_"):
                if np.ndim(model.coef_) == 1:
                    coef_values = model.coef_
                else:
                    coef_values = np.mean(np.abs(model.coef_), axis=0)
                importance = pd.DataFrame({"feature": x_encoded.columns, "importance": np.abs(coef_values)})
                importance = importance.sort_values("importance", ascending=False)
            else:
                importance = pd.DataFrame(columns=["feature", "importance"])

            st.session_state[artifact_key] = {
                "model": model,
                "resolved_mode": resolved_mode,
                "target": target,
                "features": features,
                "feature_fill_values": feature_fill_values,
                "category_maps": category_maps,
                "x_columns": x_encoded.columns.tolist(),
                "metrics": metrics,
                "importance": importance,
                "y_test": y_test,
                "y_pred": y_pred,
                "class_labels": class_labels,
                "pred_df": pd.DataFrame({"actual": y_test, "predicted": y_pred}),
                "dropped_encoded": dropped_encoded,
            }
            st.session_state[signature_key] = config_signature

        status.success("Model trained. You can now run predictions below.")

    artifact = st.session_state.get(artifact_key)
    current_signature = st.session_state.get(signature_key)
    if artifact is None:
        st.info("Configure settings and click Train / Refresh Model.")
        return
    if current_signature != config_signature:
        st.info("Configuration changed. Click Train / Refresh Model to update results.")
        return

    if artifact.get("dropped_encoded", 0) > 0:
        st.caption(
            f"Dropped {artifact['dropped_encoded']} low-frequency encoded columns "
            "to keep training responsive."
        )

    st.markdown(f"##### Evaluation ({artifact['resolved_mode']})")
    if artifact["resolved_mode"] == "Regression":
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{artifact['metrics']['mae']:,.4f}")
        m2.metric("RMSE", f"{artifact['metrics']['rmse']:,.4f}")
        m3.metric("R²", f"{artifact['metrics']['r2']:.4f}")

        fig_pred = px.scatter(
            artifact["pred_df"],
            x="actual",
            y="predicted",
            title="Actual vs Predicted",
            opacity=0.75,
        )
        fig_pred.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_pred, width="stretch")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{artifact['metrics']['accuracy']:.4f}")
        m2.metric("Precision", f"{artifact['metrics']['precision']:.4f}")
        m3.metric("Recall", f"{artifact['metrics']['recall']:.4f}")
        m4.metric("F1", f"{artifact['metrics']['f1']:.4f}")

        class_labels = artifact.get("class_labels", [])
        if len(class_labels) <= 30:
            label_ids = list(range(len(class_labels))) if class_labels else None
            cm = confusion_matrix(artifact["y_test"], artifact["y_pred"], labels=label_ids)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                x=class_labels if class_labels else None,
                y=class_labels if class_labels else None,
                color_continuous_scale=[[0, "#122038"], [1, "#2de2c4"]],
                title="Confusion Matrix",
            )
            fig_cm.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=420)
            st.plotly_chart(fig_cm, width="stretch")
        else:
            st.info("Confusion matrix hidden for high-cardinality targets (>30 classes).")

    importance = artifact.get("importance", pd.DataFrame(columns=["feature", "importance"]))
    if not importance.empty:
        st.markdown("##### Model Signals")
        top_importance = importance.head(20).sort_values("importance")
        fig_imp = px.bar(
            top_importance,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#182335", "#2de2c4"],
            title=f"Top signals from {model_name}",
        )
        fig_imp.update_layout(margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
        st.plotly_chart(fig_imp, width="stretch")

    st.markdown("##### Prediction / Classification Form")
    with st.form("sup_prediction_form"):
        input_values: dict[str, float | str] = {}
        for col in artifact["features"]:
            if pd.api.types.is_numeric_dtype(df[col]):
                default_value = float(artifact["feature_fill_values"].get(col, 0.0))
                input_values[col] = st.number_input(
                    f"{col}",
                    value=default_value,
                    step=0.1,
                    key=f"sup_form_num_{col}",
                )
            else:
                options = artifact["category_maps"].get(col, [])
                options = options[: max_category_levels]
                if "<other>" not in options:
                    options = options + ["<other>"]
                default_choice = str(artifact["feature_fill_values"].get(col, options[0]))
                if default_choice not in options:
                    default_choice = "<other>"
                input_values[col] = st.selectbox(
                    f"{col}",
                    options=options,
                    index=options.index(default_choice),
                    key=f"sup_form_cat_{col}",
                )

        submitted = st.form_submit_button("Run Model")

    if submitted:
        prediction_row = pd.DataFrame([input_values])
        for col in prediction_row.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_value = pd.to_numeric(prediction_row[col], errors="coerce")
                fallback_value = float(artifact["feature_fill_values"].get(col, 0.0))
                prediction_row[col] = numeric_value.fillna(fallback_value)
            else:
                value = prediction_row[col].astype(str)
                allowed = set(artifact["category_maps"].get(col, []))
                prediction_row[col] = value.where(value.isin(allowed), "<other>")

        row_encoded = pd.get_dummies(prediction_row, drop_first=False)
        row_encoded = row_encoded.reindex(columns=artifact["x_columns"], fill_value=0)
        pred_value = artifact["model"].predict(row_encoded)[0]

        if artifact["resolved_mode"] == "Regression":
            st.success(f"Predicted {artifact['target']}: {float(pred_value):,.4f}")
        else:
            class_labels = artifact.get("class_labels", [])
            predicted_label = class_labels[int(pred_value)] if class_labels else str(pred_value)
            st.success(f"Predicted class for {artifact['target']}: {predicted_label}")

            if hasattr(artifact["model"], "predict_proba") and class_labels:
                probs = artifact["model"].predict_proba(row_encoded)[0]
                prob_df = pd.DataFrame({"class": class_labels, "probability": probs})
                prob_df = prob_df.sort_values("probability", ascending=False).head(10)
                st.dataframe(prob_df, width="stretch", hide_index=True)


def render_time_series_tab(
    df: pd.DataFrame,
    numeric: list[str],
    datetime_candidates: list[str] | None = None,
) -> None:
    st.subheader("Time Series")
    candidates = datetime_candidates if datetime_candidates is not None else detect_datetime_candidates(df)
    if not candidates:
        st.info("No likely datetime columns detected in the current dataset/filter.")
        return

    if not numeric:
        st.info("No numeric columns available for time series values.")
        return

    d1, d2, d3 = st.columns(3)
    with d1:
        date_col = st.selectbox("Date column", candidates, key="ts_date")
    with d2:
        value_col = st.selectbox("Value column", numeric, key="ts_value")
    with d3:
        agg_name = st.selectbox("Aggregation", ["mean", "sum", "median", "max", "min"], key="ts_agg")

    freq_name = st.selectbox("Resample frequency", ["Day", "Week", "Month", "Quarter"], key="ts_freq")
    freq_map = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q"}
    freq_code = freq_map[freq_name]

    ts = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        }
    ).dropna()
    if ts.empty:
        st.warning("No valid date/value pairs after parsing.")
        return

    ts = ts.sort_values("date").set_index("date")
    series = getattr(ts["value"].resample(freq_code), agg_name)()
    if series.empty:
        st.warning("No values available after resampling.")
        return

    max_window = min(48, len(series))
    if max_window < 2:
        st.warning("Not enough aggregated points for rolling statistics.")
        return

    window = st.slider("Rolling window", 2, max_window, min(6, max_window), key="ts_window")
    rolling = series.rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=f"{agg_name} ({freq_name.lower()})",
            line=dict(color="#2de2c4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling.values,
            mode="lines",
            name=f"Rolling {window}",
            line=dict(color="#f5ae57", width=2),
        )
    )
    fig.update_layout(
        title=f"{value_col} over time",
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title="Date",
        yaxis_title=value_col,
    )
    st.plotly_chart(fig, width="stretch")


def render_insights_tab(df: pd.DataFrame, numeric: list[str]) -> None:
    st.subheader("Auto Insights")
    insights = build_auto_insights(df, numeric)
    for item in insights:
        st.markdown(f"- {item}")

    st.markdown("##### Correlation Spotlight")
    if len(numeric) >= 2:
        corr = df[numeric].corr().round(2)
        st.dataframe(corr, width="stretch")
    else:
        st.info("Add at least two numeric columns for correlation insights.")
