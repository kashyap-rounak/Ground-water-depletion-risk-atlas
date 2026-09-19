"""One-time download of India district boundaries for the choropleth map."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.geo.geo import download_boundaries, load_boundaries


def main() -> None:
    cfg = load_config()
    try:
        path = download_boundaries(cfg.geo.get("boundaries_url"), Path(cfg.geo_dir))
        print(f"District boundaries cached at: {path}")
        gdf = load_boundaries(cfg)
        if gdf is not None:
            print(f"Loaded {len(gdf)} district polygons.")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not download boundaries ({exc}). The dashboard will use the centroid fallback map.")


if __name__ == "__main__":
    main()
