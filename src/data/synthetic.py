"""Synthetic but realistic station-level groundwater + rainfall generator.

Produces data in exactly the same schema that `ingest` produces for real
files, so the pipeline runs identically on real data when you provide it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config

# A compact sample of real Indian states and districts (name, state, lat, lon,
# mean annual rainfall mm, baseline depth-to-water m, depletion m/yr).
_DISTRICTS: list[tuple[str, str, float, float, float, float, float]] = [
    ("Jodhpur", "Rajasthan", 26.29, 73.03, 350, 42.0, 0.55),
    ("Jaipur", "Rajasthan", 26.91, 75.79, 550, 30.0, 0.42),
    ("Barmer", "Rajasthan", 25.75, 71.39, 280, 48.0, 0.35),
    ("Nagaur", "Rajasthan", 27.20, 73.73, 400, 38.0, 0.40),
    ("Bikaner", "Rajasthan", 28.02, 73.31, 260, 55.0, 0.28),
    ("Ahmedabad", "Gujarat", 23.03, 72.58, 800, 22.0, 0.38),
    ("Rajkot", "Gujarat", 22.30, 70.80, 600, 28.0, 0.45),
    ("Kachchh", "Gujarat", 23.25, 69.67, 340, 40.0, 0.22),
    ("Mehsana", "Gujarat", 23.59, 72.37, 700, 35.0, 0.60),
    ("Surat", "Gujarat", 21.17, 72.83, 1200, 12.0, 0.08),
    ("Ludhiana", "Punjab", 30.90, 75.85, 700, 18.0, 0.52),
    ("Bathinda", "Punjab", 30.21, 74.94, 450, 24.0, 0.58),
    ("Amritsar", "Punjab", 31.63, 74.87, 650, 16.0, 0.35),
    ("Jalandhar", "Punjab", 31.33, 75.58, 700, 15.0, 0.40),
    ("Hisar", "Haryana", 29.15, 75.72, 450, 26.0, 0.48),
    ("Karnal", "Haryana", 29.69, 76.99, 750, 12.0, 0.30),
    ("Gurugram", "Haryana", 28.46, 77.03, 600, 30.0, 0.65),
    ("Meerut", "Uttar Pradesh", 28.98, 77.71, 800, 14.0, 0.35),
    ("Agra", "Uttar Pradesh", 27.18, 78.01, 650, 20.0, 0.42),
    ("Lucknow", "Uttar Pradesh", 26.85, 80.95, 1000, 11.0, 0.18),
    ("Varanasi", "Uttar Pradesh", 25.32, 82.97, 1050, 9.0, 0.12),
    ("Bareilly", "Uttar Pradesh", 28.37, 79.43, 1100, 8.0, 0.10),
    ("Bhopal", "Madhya Pradesh", 23.26, 77.41, 1150, 13.0, 0.20),
    ("Indore", "Madhya Pradesh", 22.72, 75.86, 950, 18.0, 0.32),
    ("Jabalpur", "Madhya Pradesh", 23.18, 79.99, 1350, 10.0, 0.10),
    ("Gwalior", "Madhya Pradesh", 26.22, 78.18, 800, 22.0, 0.38),
    ("Nashik", "Maharashtra", 19.99, 73.79, 700, 15.0, 0.28),
    ("Pune", "Maharashtra", 18.52, 73.86, 750, 14.0, 0.25),
    ("Aurangabad", "Maharashtra", 19.88, 75.34, 650, 20.0, 0.42),
    ("Nagpur", "Maharashtra", 21.15, 79.09, 1100, 9.0, 0.12),
    ("Solapur", "Maharashtra", 17.66, 75.91, 550, 25.0, 0.50),
    ("Latur", "Maharashtra", 18.40, 76.58, 600, 24.0, 0.55),
    ("Hyderabad", "Telangana", 17.39, 78.49, 900, 16.0, 0.30),
    ("Nalgonda", "Telangana", 17.05, 79.27, 750, 22.0, 0.48),
    ("Warangal", "Telangana", 17.97, 79.59, 1000, 14.0, 0.35),
    ("Medak", "Telangana", 18.05, 78.27, 900, 15.0, 0.40),
    ("Bengaluru Rural", "Karnataka", 13.23, 77.58, 850, 20.0, 0.45),
    ("Belgaum", "Karnataka", 15.85, 74.50, 1300, 8.0, 0.08),
    ("Mysuru", "Karnataka", 12.30, 76.64, 800, 12.0, 0.22),
    ("Kolar", "Karnataka", 13.14, 78.13, 700, 25.0, 0.62),
    ("Coimbatore", "Tamil Nadu", 11.02, 76.96, 700, 22.0, 0.50),
    ("Salem", "Tamil Nadu", 11.66, 78.15, 900, 18.0, 0.42),
    ("Madurai", "Tamil Nadu", 9.93, 78.12, 850, 15.0, 0.30),
    ("Tiruppur", "Tamil Nadu", 11.11, 77.34, 750, 20.0, 0.55),
    ("Chennai", "Tamil Nadu", 13.08, 80.27, 1400, 7.0, 0.15),
    ("Kolkata", "West Bengal", 22.57, 88.36, 1600, 5.0, 0.05),
    ("Barddhaman", "West Bengal", 23.24, 87.86, 1400, 6.0, 0.08),
    ("Patna", "Bihar", 25.59, 85.14, 1100, 7.0, 0.10),
    ("Gaya", "Bihar", 24.79, 85.00, 1050, 9.0, 0.20),
    ("Muzaffarpur", "Bihar", 26.12, 85.39, 1200, 6.0, 0.06),
    ("Bhubaneswar", "Odisha", 20.30, 85.82, 1500, 5.0, 0.05),
    ("Koraput", "Odisha", 18.81, 82.71, 1550, 6.0, 0.04),
    ("Raipur", "Chhattisgarh", 21.25, 81.63, 1300, 8.0, 0.10),
    ("Bilaspur", "Chhattisgarh", 22.08, 82.15, 1200, 7.0, 0.08),
    ("Ernakulam", "Kerala", 9.98, 76.30, 3000, 2.0, 0.02),
    ("Guwahati", "Assam", 26.14, 91.74, 1700, 3.0, 0.03),
    ("Ranchi", "Jharkhand", 23.34, 85.31, 1400, 10.0, 0.22),
    ("Dhanbad", "Jharkhand", 23.80, 86.44, 1300, 9.0, 0.25),
    ("Udaipur", "Rajasthan", 24.58, 73.71, 650, 20.0, 0.30),
    ("Alwar", "Rajasthan", 27.55, 76.63, 600, 25.0, 0.45),
]

_MONTHLY_RAIN_SHARES = np.array(
    [0.005, 0.008, 0.012, 0.025, 0.050, 0.150, 0.250, 0.230, 0.150, 0.080, 0.025, 0.015]
)
_MONTHLY_RAIN_SHARES = _MONTHLY_RAIN_SHARES / _MONTHLY_RAIN_SHARES.sum()

_PRE_MONSOON_MONTHS = (3, 4, 5)
_POST_MONSOON_MONTHS = (10, 11, 12)


def _season_for_month(month: int) -> str:
    if month in _PRE_MONSOON_MONTHS:
        return "pre-monsoon"
    if month in _POST_MONSOON_MONTHS:
        return "post-monsoon"
    return "other"


def generate_synthetic(cfg: Config, seed: int = 42) -> pd.DataFrame:
    """Create station-level readings for `n_districts` districts.

    Schema: district, state, station_id, date, level_mbgl, rainfall_mm,
    latitude, longitude.  Two wells per district, measured pre- and
    post-monsoon every year, with monthly rainfall rows per district.
    """
    syn = cfg.synthetic_cfg
    start_year = int(syn.get("start_year", 2000))
    end_year = int(syn.get("end_year", 2024))
    n_districts = int(syn.get("n_districts", 60))
    rng = np.random.default_rng(seed)

    chosen = _DISTRICTS[: min(n_districts, len(_DISTRICTS))]

    records: list[dict] = []
    for name, state, lat, lon, mean_rain, base_level, decline in chosen:
        for well in (1, 2):
            station_id = f"{name.replace(' ', '_')}_W{well}"
            for year in range(start_year, end_year + 1):
                for month in _PRE_MONSOON_MONTHS + _POST_MONSOON_MONTHS:
                    t_years = (year - start_year) + (month - 1) / 12.0
                    # declining water level, well-specific offset, noise, recovery
                    # after monsoon (post-monsoon levels are shallower)
                    well_offset = 0.0 if well == 1 else rng.normal(1.5, 0.4)
                    seasonal_recovery = -1.8 if month in _POST_MONSOON_MONTHS else 0.0
                    level = (
                        base_level
                        + well_offset
                        + decline * t_years
                        + seasonal_recovery
                        + rng.normal(0, 0.7)
                    )
                    # rare missing / bad values to exercise the cleaner
                    if rng.random() < 0.02:
                        level = np.nan
                    elif rng.random() < 0.004:
                        level = -7.0  # sensor error
                    rainfall = max(0.0, mean_rain * _MONTHLY_RAIN_SHARES[month - 1] * rng.gamma(2.2, 0.55))
                    records.append(
                        {
                            "district": name,
                            "state": state,
                            "station_id": station_id,
                            "date": pd.Timestamp(year=year, month=month, day=15),
                            "level_mbgl": round(float(level), 2),
                            "rainfall_mm": round(float(rainfall), 1),
                            "latitude": lat + rng.normal(0, 0.05),
                            "longitude": lon + rng.normal(0, 0.05),
                        }
                    )

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["district", "station_id", "date"]).reset_index(drop=True)
