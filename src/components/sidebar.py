import streamlit as st


def render_datavision_sidebar() -> tuple[bool, object | None]:
    """Render the DataVision sidebar and return (logout_clicked, uploaded_file)."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <h2>DataVision</h2>
            <p>Protected analytics workspace</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    logout_clicked = st.sidebar.button("Logout", width="stretch")

    st.sidebar.markdown("<div class='sidebar-section-title'>Data Input</div>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload a comma-separated dataset to start the dashboard.",
    )
    return logout_clicked, uploaded_file
