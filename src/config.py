"""Central configuration loading for the Groundwater Depletion Risk Atlas."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    """Typed view over config.yaml with convenient path helpers."""

    raw: Dict[str, Any] = field(default_factory=dict)

    # ---- paths -------------------------------------------------------
    @property
    def paths(self) -> Dict[str, str]:
        return self.raw.get("paths", {})

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / self.paths.get("processed", "data/processed")

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / self.paths.get("raw", "data/raw")

    @property
    def geo_dir(self) -> Path:
        return PROJECT_ROOT / self.paths.get("geo", "data/geo")

    @property
    def artifacts_dir(self) -> Path:
        return PROJECT_ROOT / self.paths.get("artifacts", "models/artifacts")

    @property
    def figures_dir(self) -> Path:
        return PROJECT_ROOT / self.paths.get("figures", "reports/figures")

    # ---- data --------------------------------------------------------
    @property
    def data_cfg(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def data_source(self) -> str:
        return str(self.data_cfg.get("source", "synthetic")).lower()

    @property
    def local_path(self) -> str:
        p = str(self.data_cfg.get("local_path", "") or "")
        if p:
            # Expand user paths like ~/Downloads/data
            return os.path.expanduser(p)
        return str(self.raw_dir)

    @property
    def allow_synthetic_fallback(self) -> bool:
        return bool(self.data_cfg.get("allow_synthetic_fallback", True))

    @property
    def synthetic_cfg(self) -> Dict[str, Any]:
        return self.data_cfg.get("synthetic", {})

    @property
    def canonical(self) -> Dict[str, str]:
        return self.data_cfg.get("canonical", {})

    @property
    def column_aliases(self) -> Dict[str, List[str]]:
        return self.data_cfg.get("column_aliases", {})

    @property
    def cleaning(self) -> Dict[str, Any]:
        return self.data_cfg.get("cleaning", {})

    @property
    def seasons(self) -> Dict[str, Any]:
        return self.data_cfg.get("seasons", {})

    # ---- features / models --------------------------------------------
    @property
    def features(self) -> Dict[str, Any]:
        return self.raw.get("features", {})

    @property
    def sarima(self) -> Dict[str, Any]:
        return self.raw.get("sarima", {})

    @property
    def xgboost(self) -> Dict[str, Any]:
        return self.raw.get("xgboost", {})

    @property
    def risk(self) -> Dict[str, Any]:
        return self.raw.get("risk", {})

    @property
    def shap(self) -> Dict[str, Any]:
        return self.raw.get("shap", {})

    @property
    def geo(self) -> Dict[str, Any]:
        return self.raw.get("geo", {})


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml from the project root (or an explicit path)."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw=raw)
