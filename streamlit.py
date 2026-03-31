import hmac
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="CSVDash", page_icon="📊", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

            :root {
                --bg-1: #f6f7f3;
                --bg-2: #e7efe8;
                --card: #ffffff;
                --ink: #172027;
                --muted: #5f6b72;
                --accent: #0f766e;
                --accent-2: #d97706;
                --line: #d9e0e3;
            }

            .stApp {
                background:
                    radial-gradient(1200px 500px at -10% -15%, rgba(217, 119, 6, 0.16), transparent 55%),
                    radial-gradient(1000px 450px at 105% -20%, rgba(15, 118, 110, 0.18), transparent 58%),
                    linear-gradient(180deg, var(--bg-1), var(--bg-2));
                color: var(--ink);
                font-family: 'Manrope', sans-serif;
            }

            h1, h2, h3 {
                font-family: 'Space Grotesk', sans-serif;
                letter-spacing: -0.02em;
            }

            .hero {
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 20px 24px;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.72));
                box-shadow: 0 12px 30px rgba(23, 32, 39, 0.08);
                margin-bottom: 14px;
                animation: riseIn .45s ease-out;
            }

            .metric-card {
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 10px 14px;
                background: var(--card);
                box-shadow: 0 8px 16px rgba(23, 32, 39, 0.06);
                animation: riseIn .5s ease-out;
            }

            @keyframes riseIn {
                from {
                    opacity: 0;
                    transform: translateY(8px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .stButton > button {
                border-radius: 999px;
                border: 1px solid transparent;
                font-weight: 700;
                background: linear-gradient(90deg, var(--accent), #0a5c58);
                color: #ffffff;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                border: 1px solid var(--line);
                background: rgba(255, 255, 255, 0.75);
            }

            .stDataFrame, .stPlotlyChart {
                animation: riseIn .55s ease-out;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_secret(name: str, default: str | None = None) -> str | None:
    if name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name, default)


def require_login() -> None:
    expected_password = get_secret("APP_PASSWORD")
    expected_username = get_secret("APP_USERNAME", "admin")

    if not expected_password:
        st.error(
            "Missing APP_PASSWORD. Add APP_USERNAME and APP_PASSWORD in Streamlit Cloud > App settings > Secrets."
        )
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.markdown("## Secure Access")
    st.caption("Enter credentials to open the dashboard.")

    left, center, right = st.columns([1, 1.6, 1])
    with center:
        with st.form("login", clear_on_submit=False):
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Enter Dashboard")

        if submitted:
            user_ok = hmac.compare_digest(username.strip(), expected_username)
            pass_ok = hmac.compare_digest(password, expected_password)
            if user_ok and pass_ok:
                st.session_state.authenticated = True
                st.rerun()
            st.error("Invalid credentials.")

    st.stop()


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding_errors="ignore")


def render_hero(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">CSVDash Visual Analytics</h1>
            <p style="margin:8px 0 0 0;color:#5f6b72;">
                Clean exploratory analysis workspace for large CSV files with heatmaps, relationships, and clustering.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing = int(df.isna().sum().sum())
    numeric_cols = len(df.select_dtypes(include="number").columns)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown('<div class="metric-card">', unsafe_allow_html=True)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col1.markdown("</div>", unsafe_allow_html=True)

    col2.markdown('<div class="metric-card">', unsafe_allow_html=True)
    col2.metric("Columns", df.shape[1])
    col2.markdown("</div>", unsafe_allow_html=True)

    col3.markdown('<div class="metric-card">', unsafe_allow_html=True)
    col3.metric("Numeric Features", numeric_cols)
    col3.markdown("</div>", unsafe_allow_html=True)

    col4.markdown('<div class="metric-card">', unsafe_allow_html=True)
    col4.metric("Missing Values", f"{missing:,}")
    col4.markdown("</div>", unsafe_allow_html=True)


def render_data_tab(df: pd.DataFrame) -> None:
    st.subheader("Data Overview")
    info = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing": [int(df[col].isna().sum()) for col in df.columns],
            "unique": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("##### Column Profile")
        st.dataframe(info, use_container_width=True, hide_index=True)
    with right:
        st.markdown("##### Data Preview")
        sample_size = st.slider("Preview rows", 10, 100, 25)
        st.dataframe(df.head(sample_size), use_container_width=True)

    numeric = list(df.select_dtypes(include="number").columns)
    if numeric:
        st.markdown("##### Summary Statistics")
        stats_cols = st.multiselect(
            "Select numeric columns",
            options=numeric,
            default=numeric[: min(6, len(numeric))],
        )
        if stats_cols:
            st.dataframe(df[stats_cols].describe().T, use_container_width=True)


def render_distribution_tab(df: pd.DataFrame, numeric: list[str]) -> None:
    st.subheader("Distribution Lab")
    if not numeric:
        st.warning("No numeric columns available for distribution charts.")
        return

    col1, col2 = st.columns(2)
    with col1:
        hist_col = st.selectbox("Histogram feature", numeric)
        bins = st.slider("Number of bins", 8, 120, 36)
        fig_hist = px.histogram(
            df,
            x=hist_col,
            nbins=bins,
            color_discrete_sequence=["#0f766e"],
            title=f"Distribution: {hist_col}",
        )
        fig_hist.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        box_col = st.selectbox("Box plot feature", numeric, index=min(1, len(numeric) - 1))
        fig_box = px.box(
            df,
            y=box_col,
            points="outliers",
            color_discrete_sequence=["#d97706"],
            title=f"Outliers: {box_col}",
        )
        fig_box.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(fig_box, use_container_width=True)


def render_relationship_tab(df: pd.DataFrame, numeric: list[str]) -> None:
    st.subheader("Heatmaps And Relationships")
    if len(numeric) < 2:
        st.warning("At least two numeric columns are needed for correlations and relationships.")
        return

    method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True)
    corr = df[numeric].corr(method=method, numeric_only=True)

    fig_heatmap = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"Correlation Heatmap ({method.title()})",
    )
    fig_heatmap.update_layout(margin=dict(l=10, r=10, t=52, b=10), height=560)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("##### Scatter Exploration")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X axis", numeric, key="scatter_x")
    with col2:
        y_axis = st.selectbox("Y axis", numeric, index=min(1, len(numeric) - 1), key="scatter_y")
    with col3:
        color_options = ["None"] + list(df.columns)
        color_by = st.selectbox("Color by", color_options)

    if color_by == "None":
        fig_scatter = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            opacity=0.75,
            color_discrete_sequence=["#0f766e"],
            title=f"Scatter: {x_axis} vs {y_axis}",
        )
    else:
        fig_scatter = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            opacity=0.8,
            title=f"Scatter: {x_axis} vs {y_axis}",
        )
    fig_scatter.update_layout(margin=dict(l=10, r=10, t=52, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)


def render_clustering_tab(df: pd.DataFrame, numeric: list[str]) -> None:
    st.subheader("Clustering Studio")
    if len(numeric) < 2:
        st.warning("At least two numeric columns are needed for clustering.")
        return

    features = st.multiselect(
        "Select clustering features",
        options=numeric,
        default=numeric[: min(4, len(numeric))],
    )
    if len(features) < 2:
        st.info("Select at least 2 features.")
        return

    cluster_df = df[features].dropna()
    if len(cluster_df) < 3:
        st.warning("Not enough valid rows after removing missing values.")
        return

    max_k = min(10, len(cluster_df) - 1)
    if max_k < 2:
        st.warning("Not enough observations to estimate clusters.")
        return

    k = st.slider("Number of clusters (k)", min_value=2, max_value=max_k, value=min(4, max_k))
    scale_data = st.checkbox("Standardize features", value=True)

    model_input = cluster_df.copy()
    if scale_data:
        model_input = pd.DataFrame(
            StandardScaler().fit_transform(model_input),
            columns=cluster_df.columns,
            index=cluster_df.index,
        )

    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(model_input)

    labelled = cluster_df.copy()
    labelled["cluster"] = labels.astype(str)

    quality_left, quality_right = st.columns(2)
    quality_left.metric("Inertia", f"{model.inertia_:,.2f}")
    sil_score = silhouette_score(model_input, labels)
    quality_right.metric("Silhouette", f"{sil_score:.3f}")

    if len(features) == 2:
        plot_frame = labelled.rename(columns={features[0]: "dim1", features[1]: "dim2"})
        x_title = features[0]
        y_title = features[1]
    else:
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(model_input)
        plot_frame = pd.DataFrame(reduced, columns=["dim1", "dim2"], index=model_input.index)
        plot_frame["cluster"] = labels.astype(str)
        explained = pca.explained_variance_ratio_.sum() * 100
        st.caption(f"PCA projection used for visualization ({explained:.1f}% variance explained).")
        x_title = "Component 1"
        y_title = "Component 2"

    fig_clusters = px.scatter(
        plot_frame,
        x="dim1",
        y="dim2",
        color="cluster",
        title="K-Means Cluster Projection",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_clusters.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        margin=dict(l=10, r=10, t=52, b=10),
    )
    st.plotly_chart(fig_clusters, use_container_width=True)

    st.markdown("##### Cluster Profiles")
    profile = labelled.groupby("cluster")[features].mean().round(3)
    st.dataframe(profile, use_container_width=True)


def main() -> None:
    inject_styles()
    require_login()

    st.sidebar.title("CSVDash")
    st.sidebar.caption("Protected analytics workspace")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.markdown("### Upload a CSV to begin")
        st.info("Use the file uploader in the sidebar. All analysis updates automatically after upload.")
        return

    try:
        df = load_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to parse CSV: {exc}")
        return

    if df.empty:
        st.warning("The uploaded file has no rows.")
        return

    render_hero(df)
    numeric = list(df.select_dtypes(include="number").columns)

    tab_data, tab_dist, tab_rels, tab_cluster = st.tabs(
        ["Data", "Distribution", "Heatmaps", "Clustering"]
    )
    with tab_data:
        render_data_tab(df)
    with tab_dist:
        render_distribution_tab(df, numeric)
    with tab_rels:
        render_relationship_tab(df, numeric)
    with tab_cluster:
        render_clustering_tab(df, numeric)


if __name__ == "__main__":
    main()
