#!/usr/bin/env python3
"""
Script Name: raob_helpers.py
Purpose: European radiosonde (RAOB) batch fetch and 500 hPa level extraction.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-04-15
Version: 1.2.0
License: MIT

Notes:
  • Context: Processing and field-assembly module for upper-air data. Analogous
             to contouring_helpers for surface fields. Internally fetches the
             IGRA2 station list, then batch-fetches 500 hPa soundings via
             wyoming_raob using ThreadPoolExecutor (~130 European sites).
             Output is a station DataFrame ready for the plotting layer.
             Stations with no available sounding are silently skipped.
  • Inputs:  Bundled IGRA2 station list at data/igra2-station-list.txt
             (no network access required for station metadata).
             Sounding data acquired at runtime from the University of Wyoming
             upper-air archive.
  • Outputs: pd.DataFrame with one row per station containing 500 hPa
             geopotential height, wind components, temperature, and dew-point.
  • Configuration: MAX_RAOB_WORKERS, RAOB_MAX_RETRIES, RAOB_RETRY_BASE_S,
                   _IGRA2_STATION_LIST_PATH.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Local
from wyoming_raob import fetch_latest_sounding, fetch_retrospective_sounding

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MAX_RAOB_WORKERS   = 16   # parallel threads for batch sounding fetch
RAOB_MAX_RETRIES   = 3    # retry attempts on 503 / transient server errors
RAOB_RETRY_BASE_S  = 2.0  # back-off base (seconds); actual = BASE * 2^(attempt-1)

# Path to the bundled IGRA2 station list (relative to this file).
_IGRA2_STATION_LIST_PATH = Path(__file__).parent.parent / "data" / "igra2-station-list.txt"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("owl.raob.helpers")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Station list]


def fetch_igra2_europe(
    lat_min: float = 30.0,
    lat_max: float = 72.0,
    lon_min: float = -25.0,
    lon_max: float = 45.0,
    path: Path = _IGRA2_STATION_LIST_PATH,
) -> pd.DataFrame:
    """
    Read the bundled IGRA2 station list and return European sites.

    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds in degrees North (default 30 – 72).
    lon_min, lon_max : float
        Longitude bounds in degrees East (default -25 – 45).
    path : Path
        Local path to the IGRA2 station-list text file
        (default: ``data/igra2-station-list.txt`` in the project root).

    Returns
    -------
    pd.DataFrame
        Columns:

        ============== =============================================
        wmo_id         WMO station number as a string (e.g. '16622')
        name           Station name (e.g. 'ATHINAI (HELLINIKON)')
        latitude_deg   Latitude in degrees North
        longitude_deg  Longitude in degrees East
        elevation_m    Elevation in metres above MSL
        ============== =============================================
    Reference(s)
    ----------
    Durre, I., Yin, X., Vose, R.S., Applequist, S., Arnfield J.,
    Korzeniewski, B. & Hundermark, B. (2016).
    Integrated Global Radiosonde Archive (IGRA), Version 2.
    NOAA National Centers for Environmental Information.
    https://doi.org/10.7289/V5X63K0Q
    """
    raw_text = Path(path).read_text(encoding="utf-8")

    records = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=4)
        if len(parts) < 4:
            continue
        code = parts[0]
        try:
            lat = float(parts[1])
            lon = float(parts[2])
            elev = float(parts[3])
        except ValueError:
            continue
        name = parts[4].strip() if len(parts) == 5 else ""

        # WMO ID: last 5 characters of the IGRA2 code.
        # e.g. GRM00016622 → '16622', SPM00008221 → '08221'
        # Using code[-5:] preserves leading zeros (e.g. 08xxx for Iberia)
        # that str(int(...)) would silently strip.
        # Non-WMO network codes have alphanumeric site numbers — skip them.
        wmo_id = code[-5:]
        if not wmo_id.isdigit():
            continue

        records.append({
            "wmo_id":        wmo_id,
            "name":          name,
            "latitude_deg":  lat,
            "longitude_deg": lon,
            "elevation_m":   elev,
        })

    df = pd.DataFrame(records)
    mask = (
        (df["latitude_deg"]  >= lat_min) & (df["latitude_deg"]  <= lat_max) &
        (df["longitude_deg"] >= lon_min) & (df["longitude_deg"] <= lon_max)
    )
    df = df[mask].reset_index(drop=True)
    LOG.info("[IGRA2] %d European radiosonde stations in bounding box", len(df))
    return df


# [Internal helpers]


def _nearest_sounding_time(date_str: str, hour: int) -> tuple[str, int]:
    """
    Map an arbitrary target UTC hour to the nearest canonical sounding time.

    Radiosondes are typically launched at 00 and 12 UTC, so any target hour
    is snapped to the nearest of those, crossing the date boundary when the
    target is closer to 00 UTC of the following day.

    Rule:
      hour in [0, 5]    -> (date,       00)
      hour in [6, 17]   -> (date,       12)
      hour in [18, 23]  -> (date + 1,   00)
    """
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be in [0, 23], got {hour}")

    base = datetime.strptime(date_str, "%Y-%m-%d")
    if hour < 6:
        return base.strftime("%Y-%m-%d"), 0
    if hour < 18:
        return base.strftime("%Y-%m-%d"), 12
    return (base + timedelta(days=1)).strftime("%Y-%m-%d"), 0


def _extract_level(df_snd: pd.DataFrame, pressure_hpa: float, column: str) -> float:
    """
    Interpolate column from a sounding DataFrame at pressure_hpa.

    Parameters
    ----------
    df_snd : pd.DataFrame
        Sounding data with at least ``pressure`` and *column* columns.
    pressure_hpa : float
        Target pressure level (hPa).
    column : str
        Column name to interpolate.

    Returns
    -------
    float
        Interpolated value, or ``np.nan`` if data are insufficient.
    """
    s = df_snd[["pressure", column]].dropna().sort_values("pressure")
    if len(s) < 2:
        return np.nan
    pvals = s["pressure"].values
    cvals = s[column].values
    if pressure_hpa < pvals[0] or pressure_hpa > pvals[-1]:
        return np.nan
    return float(np.interp(pressure_hpa, pvals, cvals))


# [Core batch fetch]


def fetch_europe_raob_fields(
    station_meta: pd.DataFrame,
    date_str: str | None = None,
    hour: int | None = None,
) -> pd.DataFrame:
    """
    Fetch soundings for every station in station_meta and extract 500 hPa
    fields by linear interpolation on the pressure coordinate.

    Requests are issued in parallel (up to ``MAX_RAOB_WORKERS`` threads).  Stations for which
    the Wyoming archive returns no sounding are silently skipped.

    Parameters
    ----------
    station_meta : pd.DataFrame
        Output of ``fetch_igra2_europe()``.  Must contain columns
        ``wmo_id``, ``latitude_deg``, ``longitude_deg``.
    date_str : str, optional
        Target date as ``'YYYY-MM-DD'``. When combined with ``hour``, selects
        retrospective mode and the sounding closest to
        ``{date_str}T{hour}:00 UTC`` (snapped to the nearest 
        sounding hour, 00 or 12 UTC, crossing dates if necessary).
        If ``None``, the latest available sounding is fetched per station.
    hour : int, optional
        Target UTC hour (0-23). Only used when ``date_str`` is also given.

    Returns
    -------
    pd.DataFrame
        One row per successful station, columns:
        ``station_id`` (WMO number), ``longitude``, ``latitude``,
        ``z500`` (m), ``z500_dam`` (dam),
        ``u500`` (knots), ``v500`` (knots), ``wspd500`` (knots), ``wdir500`` (°),
        ``t500`` (°C), ``td500`` (°C), ``dd500`` (°C), ``valid`` (datetime).
    """
    retro_mode = date_str is not None and hour is not None
    if retro_mode:
        sounding_date, sounding_hour = _nearest_sounding_time(date_str, hour)
        LOG.info(
            "[RAOB] Retrospective mode — using %s %02d:00 UTC soundings "
            "(nearest hour to %s %02d:00 UTC)",
            sounding_date, sounding_hour, date_str, hour,
        )

    def _fetch_one(row: pd.Series):
        wmo_id = row["wmo_id"]
        for attempt in range(1, RAOB_MAX_RETRIES + 1):
            try:
                if retro_mode:
                    df_snd, valid_dt = fetch_retrospective_sounding(
                        wmo_id, sounding_date, sounding_hour,
                    )
                else:
                    df_snd, valid_dt = fetch_latest_sounding(wmo_id)
                z500  = _extract_level(df_snd, 500, "height")
                t500  = _extract_level(df_snd, 500, "temperature")
                td500 = _extract_level(df_snd, 500, "dewpoint")
                return {
                    "station_id": wmo_id,
                    "longitude":  row["longitude_deg"],
                    "latitude":   row["latitude_deg"],
                    "z500":    z500,
                    "z500_dam": z500 / 10.0 if not np.isnan(z500) else np.nan,
                    "u500":    _extract_level(df_snd, 500, "u_wind"),
                    "v500":    _extract_level(df_snd, 500, "v_wind"),
                    "wspd500": _extract_level(df_snd, 500, "wind_speed"),
                    "wdir500": _extract_level(df_snd, 500, "wind_direction"),
                    "t500":  t500,
                    "td500": td500,
                    "dd500": (t500 - td500)
                             if not (np.isnan(t500) or np.isnan(td500))
                             else np.nan,
                    "valid":   valid_dt,
                }
            except Exception as e:
                msg = str(e)
                # "Can't get" = station not in Wyoming archive — skip silently.
                if "Can't get" in msg or "no data available" in msg:
                    LOG.debug("No sounding data for %s (not in Wyoming archive)", wmo_id)
                    return None
                # 503 / transient server error — retry with back-off
                if "503" in msg and attempt < RAOB_MAX_RETRIES:
                    time.sleep(RAOB_RETRY_BASE_S * (2 ** (attempt - 1)))
                    continue
                # Any other error (404, parse error, etc.) — skip
                LOG.debug("Skipping %s: %s", wmo_id, msg)
                return None
        return None

    n_total = len(station_meta)
    results = []
    with ThreadPoolExecutor(max_workers=MAX_RAOB_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, row): row for _, row in station_meta.iterrows()}
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                results.append(row)

    LOG.info("[RAOB] %d / %d stations with available sounding", len(results), n_total)

    if not results:
        return pd.DataFrame(columns=[
            "station_id", "longitude", "latitude",
            "z500", "z500_dam", "u500", "v500", "wspd500", "wdir500",
            "t500", "td500", "dd500", "valid",
        ])

    return pd.DataFrame(results).sort_values("station_id").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Example:")
    print("  from raob_helpers import fetch_europe_raob_fields")
