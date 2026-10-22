"""Unit tests for the dashboard authentication module."""

from __future__ import annotations

from unittest.mock import patch
import pytest

from src.config import Config
from src.app.auth import check_auth


# Test whether access is allowed when authentication is disabled.
def test_auth_disabled_allows_access():
    cfg = Config(raw={"auth": {"enabled": False}})

    # When authentication is disabled, check_auth must return True immediately.
    assert check_auth(cfg) is True


# Test whether an already authenticated user is allowed to access the dashboard.
def test_auth_enabled_authenticated_session():
    cfg = Config(raw={"auth": {"enabled": True, "users": {"admin": "secret"}}})

    # Create a mock session representing an authenticated admin user.
    mock_session = {"authenticated": True, "username": "admin"}

    # Mock Streamlit session state and sidebar to avoid actual UI interactions.
    with patch("streamlit.session_state", mock_session), patch("streamlit.sidebar"):
        # Verify that the authenticated user is granted access.
        assert check_auth(cfg) is True


# Test whether an unauthenticated user is denied access to the dashboard.
def test_auth_enabled_unauthenticated_session():
    cfg = Config(raw={"auth": {"enabled": True, "users": {"admin": "secret"}}})

    # Create a mock session representing an unauthenticated user.
    mock_session = {"authenticated": False}

    # Mock Streamlit columns to prevent rendering the actual login layout.
    with patch("streamlit.session_state", mock_session), patch("streamlit.columns") as mock_cols:
        mock_cols.return_value = (None, None, None)

        # Mock the login gate to verify that it is called.
        with patch("src.app.auth.render_login_gate") as mock_gate:

            # Verify that access is denied to the unauthenticated user.
            assert check_auth(cfg) is False

            # Verify that the login gate is displayed exactly once.
            mock_gate.assert_called_once()