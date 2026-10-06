"""Tests for flexible local-file ingestion."""
from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.data.ingest import _coerce_types, _map_columns, _seasonal_wide_to_long


def test_cgwb_year_and_season_columns_convert_to_long_rows():
    cfg = load_config()
    raw = pd.DataFrame({
        "District Name": ["Example"],
        "State": ["Example State"],
        "Station Code": ["W-1"],
        "Year": [2020],
        "Pre-Monsoon": ["12.4"],
        "Post-Monsoon": ["10.7"],
    })

    mapped = _map_columns(raw, cfg)
    typed = _coerce_types(mapped)
    result = _seasonal_wide_to_long(typed)

    assert result["level_mbgl"].tolist() == [12.4, 10.7]
    assert result["date"].dt.month.tolist() == [5, 11]
    assert result["date"].dt.year.tolist() == [2020, 2020]
    assert result["season"].tolist() == ["pre-monsoon", "post-monsoon"]
    assert result["station_id"].tolist() == ["W-1", "W-1"]


def test_seasonal_wide_rows_require_a_date_or_year():
    frame = pd.DataFrame({"Pre-Monsoon": [10.0]})

    try:
        _seasonal_wide_to_long(frame)
    except ValueError as exc:
        assert "valid date or year" in str(exc)
    else:
        raise AssertionError("expected invalid seasonal input to be rejected")
