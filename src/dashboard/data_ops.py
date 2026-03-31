from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding_errors="ignore")


@st.cache_data(show_spinner=False)
def detect_datetime_candidates(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append(col)
            continue

        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        sample = series.dropna().astype(str).head(800)
        if sample.empty:
            continue

        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.75:
            candidates.append(col)

    return candidates


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    return numeric, categorical


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("<div class='sidebar-section-title'>Filters</div>", unsafe_allow_html=True)
    selected = st.sidebar.multiselect(
        "Columns to filter",
        options=df.columns.tolist(),
        help="Add one or more columns to create interactive filters.",
    )

    if not selected:
        st.sidebar.caption(f"Rows active: {len(df):,}")
        return df

    filtered = df.copy()
    for col in selected:
        series = filtered[col]

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if clean.empty:
                continue

            min_val = float(clean.min())
            max_val = float(clean.max())
            if min_val == max_val:
                st.sidebar.caption(f"{col}: fixed value {min_val:.3f}")
                continue

            low, high = st.sidebar.slider(
                f"{col} range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
            )
            filtered = filtered[filtered[col].between(low, high, inclusive="both")]
            continue

        values = series.dropna().astype(str)
        if values.empty:
            continue

        options = sorted(values.unique().tolist())
        if len(options) > 120:
            token = st.sidebar.text_input(
                f"{col} contains",
                value="",
                help="High-cardinality column detected. Filter rows by text match.",
            ).strip()
            if token:
                filtered = filtered[filtered[col].astype(str).str.contains(token, case=False, na=False)]
            continue

        chosen = st.sidebar.multiselect(
            f"{col} values",
            options=options,
            default=options,
        )
        if chosen:
            filtered = filtered[filtered[col].astype(str).isin(chosen)]
        else:
            filtered = filtered.iloc[0:0]

    st.sidebar.caption(f"Rows active: {len(filtered):,}")
    return filtered


def build_outlier_table(df: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    if not numeric:
        return pd.DataFrame(columns=["feature", "outliers", "outlier_pct", "lower_bound", "upper_bound"])

    num_df = df[numeric]
    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (num_df.lt(lower, axis=1)) | (num_df.gt(upper, axis=1))
    counts = mask.sum().astype(int)
    denom = num_df.notna().sum().replace(0, np.nan)
    pct = (counts / denom * 100).round(2).fillna(0)

    outlier_table = pd.DataFrame(
        {
            "feature": counts.index,
            "outliers": counts.values,
            "outlier_pct": pct.values,
            "lower_bound": lower.round(3).values,
            "upper_bound": upper.round(3).values,
        }
    ).sort_values("outlier_pct", ascending=False)
    return outlier_table


def build_auto_insights(df: pd.DataFrame, numeric: list[str]) -> list[str]:
    insights: list[str] = []
    insights.append(f"Dataset has {len(df):,} rows and {df.shape[1]} columns.")

    missing = df.isna().mean().sort_values(ascending=False)
    top_missing = missing[missing > 0].head(3)
    if top_missing.empty:
        insights.append("No missing values detected.")
    else:
        miss_text = ", ".join([f"{column} ({rate * 100:.1f}%)" for column, rate in top_missing.items()])
        insights.append(f"Highest missing-rate features: {miss_text}.")

    if len(numeric) >= 2:
        corr = df[numeric].corr()
        pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index(name="corr")
        if not pairs.empty:
            best = pairs.iloc[pairs["corr"].abs().argmax()]
            insights.append(
                f"Strongest linear relationship: {best['level_0']} vs {best['level_1']} (r={best['corr']:.2f})."
            )

        outliers = build_outlier_table(df, numeric)
        if not outliers.empty:
            top = outliers.iloc[0]
            insights.append(
                f"Feature with highest outlier rate: {top['feature']} ({top['outlier_pct']:.2f}% of non-null values)."
            )

    dup = int(df.duplicated().sum())
    if dup > 0:
        insights.append(f"Dataset contains {dup:,} duplicated rows.")

    return insights
