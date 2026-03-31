import hmac
import os
import re
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="CSVDash - Protected",
    page_icon=":bar_chart:",
    layout="wide",
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
            "Missing APP_PASSWORD secret. Add APP_PASSWORD in Streamlit Cloud > Advanced settings > Secrets."
        )
        st.stop()

    if st.session_state.get("authenticated", False):
        return

    st.title("CSVDash Login")
    st.caption("Protected access for your school project deployment")

    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        user_ok = hmac.compare_digest(username.strip(), expected_username)
        pass_ok = hmac.compare_digest(password, expected_password)
        if user_ok and pass_ok:
            st.session_state.authenticated = True
            st.rerun()
        st.error("Invalid username or password.")

    st.stop()


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


def find_column_name(user_text: str, columns: list[str]) -> str | None:
    cleaned = user_text.strip().lower()

    for col in columns:
        if cleaned == col.lower():
            return col

    no_space = re.sub(r"[\s_]+", "", cleaned)
    for col in columns:
        if no_space == re.sub(r"[\s_]+", "", col.lower()):
            return col

    for col in columns:
        if cleaned in col.lower():
            return col

    return None


def run_sql_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    connection = sqlite3.connect(":memory:")
    try:
        df.to_sql("dados", connection, if_exists="replace", index=False)
        result = pd.read_sql_query(query, connection)
        return result
    finally:
        connection.close()


def parse_command(command: str, df: pd.DataFrame) -> dict:
    text = command.strip()
    low = text.lower()

    if not text:
        return {"type": "empty"}

    if low.startswith("sql:"):
        return {"type": "sql", "query": text.split(":", 1)[1].strip()}

    if re.match(r"^\s*(select|with)\b", text, flags=re.IGNORECASE):
        return {"type": "sql", "query": text}

    if "estat" in low:
        return {"type": "stats"}

    top_match = re.search(r"top\s+(\d+)\s+(?:de\s+)?(.+)$", low)
    if top_match:
        n_value = int(top_match.group(1))
        col_text = top_match.group(2).strip()
        col_name = find_column_name(col_text, list(df.columns))
        if col_name:
            query = (
                f'SELECT "{col_name}", COUNT(*) AS total '
                f'FROM dados GROUP BY "{col_name}" ORDER BY total DESC LIMIT {n_value}'
            )
            return {"type": "sql", "query": query}

    mean_match = re.search(r"(?:media|m[eé]dia)\s+(?:da|de)\s+(.+)$", low)
    if mean_match:
        col_text = mean_match.group(1).replace("coluna", "").strip()
        col_name = find_column_name(col_text, list(df.columns))
        if col_name:
            query = f'SELECT AVG("{col_name}") AS media FROM dados'
            return {"type": "sql", "query": query}

    count_match = re.search(r"contagem\s+por\s+(.+)$", low)
    if count_match:
        col_text = count_match.group(1).strip()
        col_name = find_column_name(col_text, list(df.columns))
        if col_name:
            query = (
                f'SELECT "{col_name}", COUNT(*) AS total '
                f'FROM dados GROUP BY "{col_name}" ORDER BY total DESC'
            )
            return {"type": "sql", "query": query}

    scatter_match = re.search(
        r"(?:grafico|gr[aá]fico)\s+de\s+dispers[aã]o\s+(.+?)\s+(?:vs|x|e)\s+(.+)$",
        low,
    )
    if scatter_match:
        x_col_text = scatter_match.group(1).strip()
        y_col_text = scatter_match.group(2).strip()
        x_col = find_column_name(x_col_text, list(df.columns))
        y_col = find_column_name(y_col_text, list(df.columns))
        if x_col and y_col:
            return {"type": "scatter", "x": x_col, "y": y_col}

    return {"type": "unknown"}


def render_dataset_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isna().sum().sum()))

    with st.expander("Columns and data types", expanded=False):
        info_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(t) for t in df.dtypes],
                "missing": [int(df[c].isna().sum()) for c in df.columns],
                "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
            }
        )
        st.dataframe(info_df, use_container_width=True)

    st.dataframe(df.head(25), use_container_width=True)


def render_statistics(df: pd.DataFrame) -> None:
    st.subheader("Statistics")
    numeric_columns = list(df.select_dtypes(include="number").columns)

    if not numeric_columns:
        st.warning("No numeric columns detected for statistical analysis.")
        return

    selected = st.multiselect(
        "Choose numeric columns",
        numeric_columns,
        default=numeric_columns[: min(6, len(numeric_columns))],
    )
    if not selected:
        st.info("Select at least one numeric column.")
        return

    stats_df = df[selected].describe().T
    st.dataframe(stats_df, use_container_width=True)


def render_visualizations(df: pd.DataFrame) -> None:
    st.subheader("Visualizations")
    all_columns = list(df.columns)
    numeric_columns = list(df.select_dtypes(include="number").columns)

    if not numeric_columns:
        st.warning("No numeric columns found for charts.")
        return

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        hist_col = st.selectbox("Histogram column", numeric_columns, key="hist_col")
        bins = st.slider("Histogram bins", 10, 120, 35, key="hist_bins")
        fig_hist = px.histogram(df, x=hist_col, nbins=bins, title=f"Histogram - {hist_col}")
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart_col_2:
        box_col = st.selectbox("Box plot column", numeric_columns, key="box_col")
        fig_box = px.box(df, y=box_col, title=f"Box plot - {box_col}")
        st.plotly_chart(fig_box, use_container_width=True)

    if len(numeric_columns) >= 2:
        st.markdown("#### Correlation Heatmap")
        corr = df[numeric_columns].corr(numeric_only=True)
        fig_heatmap = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            origin="lower",
            title="Numeric Correlation Matrix",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("#### Scatter Plot")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            x_col = st.selectbox("X axis", numeric_columns, key="scatter_x")
        with sc2:
            y_col = st.selectbox("Y axis", numeric_columns, index=min(1, len(numeric_columns) - 1), key="scatter_y")
        with sc3:
            color_col = st.selectbox("Color by", ["(none)"] + all_columns, key="scatter_color")

        if color_col == "(none)":
            fig_scatter = px.scatter(df, x=x_col, y=y_col, title=f"Scatter - {x_col} vs {y_col}")
        else:
            fig_scatter = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"Scatter - {x_col} vs {y_col}",
            )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### K-Means Clustering")
    cluster_columns = st.multiselect(
        "Select numeric columns for clustering",
        numeric_columns,
        default=numeric_columns[: min(3, len(numeric_columns))],
        key="cluster_cols",
    )

    if len(cluster_columns) < 2:
        st.info("Select at least 2 numeric columns for clustering.")
        return

    valid_cluster_df = df[cluster_columns].dropna()
    if valid_cluster_df.empty:
        st.warning("No rows left after dropping missing values in selected cluster columns.")
        return

    max_k = min(10, len(valid_cluster_df))
    if max_k < 2:
        st.warning("Not enough rows for clustering.")
        return

    k_value = st.slider("Number of clusters (k)", 2, max_k, min(4, max_k), key="k_value")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(valid_cluster_df)

    model = KMeans(n_clusters=k_value, random_state=42, n_init="auto")
    labels = model.fit_predict(scaled)

    plot_df = valid_cluster_df.copy()
    plot_df["cluster"] = labels.astype(str)

    x_cluster = st.selectbox("Cluster X axis", cluster_columns, key="cluster_x")
    y_cluster = st.selectbox(
        "Cluster Y axis",
        cluster_columns,
        index=min(1, len(cluster_columns) - 1),
        key="cluster_y",
    )

    fig_cluster = px.scatter(
        plot_df,
        x=x_cluster,
        y=y_cluster,
        color="cluster",
        title="K-Means Cluster Projection",
    )
    st.plotly_chart(fig_cluster, use_container_width=True)


def render_chatbot_sql(df: pd.DataFrame) -> None:
    st.subheader("Chatbot Commands (PT/EN) + SQL")
    st.write("Use plain commands or direct SQL.")
    st.code(
        "\n".join(
            [
                "mostre as estatisticas",
                "media da coluna score",
                "contagem por genero",
                "grafico de dispersao idade vs salario",
                "sql: SELECT * FROM dados LIMIT 20",
            ]
        ),
        language="text",
    )

    command = st.text_input("Command")
    run = st.button("Run command")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if run:
        action = parse_command(command, df)
        st.session_state.chat_history.append({"command": command, "action": action})

        if action["type"] == "empty":
            st.info("Write a command first.")

        elif action["type"] == "stats":
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        elif action["type"] == "sql":
            try:
                result = run_sql_query(df, action["query"])
                st.success("SQL executed successfully.")
                st.dataframe(result, use_container_width=True)
            except Exception as exc:
                st.error(f"SQL error: {exc}")

        elif action["type"] == "scatter":
            fig = px.scatter(df, x=action["x"], y=action["y"], title="Command Scatter")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Unknown command. Try one of the examples shown above.")

    if st.session_state.chat_history:
        with st.expander("Command history", expanded=False):
            for i, item in enumerate(reversed(st.session_state.chat_history), start=1):
                st.write(f"{i}. {item['command']}")


def main() -> None:
    require_login()

    st.sidebar.title("CSVDash")
    st.sidebar.write("Authenticated")

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    st.title("CSV Analytics + Command Chatbot")
    st.caption("Upload a CSV and run exploratory analysis, charts, clusters, and SQL commands.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to start.")
        return

    try:
        df = load_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    tabs = st.tabs(["Overview", "Statistics", "Visualizations", "Chatbot SQL"])

    with tabs[0]:
        render_dataset_overview(df)

    with tabs[1]:
        render_statistics(df)

    with tabs[2]:
        render_visualizations(df)

    with tabs[3]:
        render_chatbot_sql(df)


if __name__ == "__main__":
    main()
