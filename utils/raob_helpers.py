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
  • Inputs:  No file inputs. Data acquired at runtime from the IGRA2 station
             list (NCEI) and the University of Wyoming upper-air archive.
  • Outputs: pd.DataFrame with one row per station containing 500 hPa
             geopotential height, wind components, temperature, and dew-point.
  • Configuration: MAX_RAOB_WORKERS, RAOB_MAX_RETRIES, RAOB_RETRY_BASE_S,
                   IGRA2_STATION_LIST_URL.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

# Local
from wyoming_raob import fetch_latest_sounding

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MAX_RAOB_WORKERS   = 16   # parallel threads for batch sounding fetch
RAOB_MAX_RETRIES   = 3    # retry attempts on 503 / transient server errors
RAOB_RETRY_BASE_S  = 2.0  # back-off base (seconds); actual = BASE * 2^(attempt-1)

IGRA2_STATION_LIST_URL = (
    "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
)

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
    url: str = IGRA2_STATION_LIST_URL,
) -> pd.DataFrame:
    """
    Fetch the IGRA2 radiosonde station list and return European sites.

    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds in degrees North (default 30 – 72).
    lon_min, lon_max : float
        Longitude bounds in degrees East (default -25 – 45).
    url : str
        URL of the IGRA2 station list text file.

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
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    records = []
    for line in resp.text.splitlines():
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
) -> pd.DataFrame:
    """
    Fetch the latest sounding for every station in station_meta and extract
    500 hPa fields by linear interpolation on the pressure coordinate.

    Requests are issued in parallel (up to ``MAX_RAOB_WORKERS`` threads).  Stations for which
    the Wyoming archive returns no sounding are silently skipped.

    Parameters
    ----------
    station_meta : pd.DataFrame
        Output of ``fetch_igra2_europe()``.  Must contain columns
        ``wmo_id``, ``latitude_deg``, ``longitude_deg``.

    Returns
    -------
    pd.DataFrame
        One row per successful station, columns:
        ``station_id`` (WMO number), ``longitude``, ``latitude``,
        ``z500`` (m), ``z500_dam`` (dam),
        ``u500`` (m/s), ``v500`` (m/s), ``wspd500`` (m/s), ``wdir500`` (°),
        ``t500`` (°C), ``td500`` (°C), ``dd500`` (°C), ``valid`` (datetime).
    """

    def _fetch_one(row: pd.Series):
        wmo_id = row["wmo_id"]
        for attempt in range(1, RAOB_MAX_RETRIES + 1):
            try:
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
