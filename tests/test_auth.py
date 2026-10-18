"""Unit tests for the dashboard authentication module."""
from __future__ import annotations

from unittest.mock import patch
import pytest

from src.config import Config
from src.app.auth import check_auth


def test_auth_disabled_allows_access():
    cfg = Config(raw={"auth": {"enabled": False}})
    # When auth is disabled, check_auth must return True immediately
    assert check_auth(cfg) is True


def test_auth_enabled_authenticated_session():
    cfg = Config(raw={"auth": {"enabled": True, "users": {"admin": "secret"}}})
    mock_session = {"authenticated": True, "username": "admin"}
    with patch("streamlit.session_state", mock_session), patch("streamlit.sidebar"):
        assert check_auth(cfg) is True


def test_auth_enabled_unauthenticated_session():
    cfg = Config(raw={"auth": {"enabled": True, "users": {"admin": "secret"}}})
    mock_session = {"authenticated": False}
    with patch("streamlit.session_state", mock_session), patch("streamlit.columns") as mock_cols:
        mock_cols.return_value = (None, None, None)
        with patch("src.app.auth.render_login_gate") as mock_gate:
            assert check_auth(cfg) is False
            mock_gate.assert_called_once()
