"""Authentication module for the Streamlit dashboard.

Supports optional authentication controlled via `config.yaml`:
    auth:
      enabled: true/false
      users:
        admin: "admin123"
        viewer: "atlas123"
"""
from __future__ import annotations

import streamlit as st

from src.config import Config


def check_auth(cfg: Config) -> bool:
    """Evaluate and render authentication if enabled in configuration.

    Returns:
        bool: True if user is authorized to view the dashboard, False otherwise.
    """
    auth_cfg = cfg.auth
    if not auth_cfg.get("enabled", False):
        return True

    # Check if already authenticated in this session
    if st.session_state.get("authenticated", False):
        user = st.session_state.get("username", "User")
        with st.sidebar:
            st.markdown(f"👤 **Logged in as:** `{user}`")
            if st.button("🚪 Log out", key="logout_btn", use_container_width=True):
                st.session_state["authenticated"] = False
                st.session_state["username"] = None
                st.rerun()
            st.divider()
        return True

    # User is not authenticated -> Render Login Gate
    render_login_gate(auth_cfg.get("users", {}))
    return False


def render_login_gate(users: dict[str, str]) -> None:
    """Render a clean login modal / container when authentication is required."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🔒 Groundwater Risk Atlas Login")
        st.caption("This portal is protected. Please sign in with your credentials to proceed.")

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", key="login_username").strip()
            password = st.text_input("Password", type="password", key="login_password").strip()
            submit = st.form_submit_button("Sign In", use_container_width=True)

            if submit:
                if not username or not password:
                    st.error("Please enter both username and password.")
                elif username in users and str(users[username]) == password:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.info("💡 **Demo accounts configured:** `admin` (`admin123`) or `viewer` (`atlas123`)")
