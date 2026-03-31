from __future__ import annotations

import hmac
import os

import streamlit as st


def get_secret(name: str, default: str | None = None) -> str | None:
    if name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name, default)


def get_bool_secret(*names: str, default: bool) -> bool:
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}

    for name in names:
        value = get_secret(name)
        if value is None:
            continue
        normalized = value.strip().lower()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False

    return default


def require_login() -> None:
    show_password_gate = get_bool_secret("SHOW_PASSWORD", default=True)
    if not show_password_gate:
        return

    expected_password = get_secret("APP_PASSWORD")
    expected_username = get_secret("APP_USERNAME", "admin")

    if not expected_password:
        st.error("Missing APP_PASSWORD. Add APP_USERNAME and APP_PASSWORD in Streamlit Cloud Secrets.")
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.markdown('<div class="login-view-marker"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <section class="login-copy">
            <h1>DataVision Secure Access</h1>
            <p>Enter credentials to unlock the analytics workspace.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1.0, 1.25, 1.0])
    with center:
        with st.form("login", clear_on_submit=False):
            username = st.text_input("Username", value=expected_username)
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
