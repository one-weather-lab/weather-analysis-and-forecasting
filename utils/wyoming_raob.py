#!/usr/bin/env python3
"""
Script Name: wyoming_raob.py
Purpose: Radiosonde data acquisition from the University of Wyoming database.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-04-15
Version: 1.2.0
License: MIT

Notes:
  • Context: Fetching module for radiosonde data. Analogous to iem_raw and
             noaa_realtime for surface data. Implements a primary cgi-bin
             endpoint with automatic WSGI/BUFR fallback for stations where
             the primary returns no data.
  • Inputs:  WMO station number (str), date/hour for retrospective requests.
  • Outputs: pd.DataFrame of sounding data.
  • Configuration: Base URLs for the primary cgi-bin and fallback WSGI/BUFR
                   endpoints. Unit conversion factor (m/s → knots). All fetch
                   parameters (station, date, hour) are exposed as function
                   arguments.

  Data sources
  ------------
  Primary — University of Wyoming cgi-bin archive:
    Endpoint: http://weather.uwyo.edu/cgi-bin/sounding
    Parameters: region=europe, TYPE=TEXT:LIST, YEAR, MONTH, FROM/TO (DDHH),
                STNM (WMO station number)
    Wind column: SKNT (knots)
    Example: http://weather.uwyo.edu/cgi-bin/sounding?region=europe&TYPE=TEXT:LIST&YEAR=2026&MONTH=04&FROM=1200&TO=1200&STNM=16716

  Fallback — University of Wyoming WSGI/BUFR archive:
    Used automatically when the primary returns "Can't get".
    Endpoint: http://weather.uwyo.edu/wsgi/sounding
    Parameters: datetime (YYYY-MM-DD HH:MM:SS), id (WMO), src=BUFR,
                type=TEXT:LIST
    Wind column: SPED (m/s) — converted to knots internally
    Example: http://weather.uwyo.edu/wsgi/sounding?datetime=2026-04-14%2000:00:00&id=08536&src=BUFR&type=TEXT:LIST

  Column mapping (TEXT:LIST → DataFrame)
  ---------------------------------------
  PRES → pressure       (hPa)
  HGHT → height         (m)
  TEMP → temperature    (°C)
  DWPT → dewpoint       (°C)
  DRCT → wind_direction (°)
  SKNT → wind_speed     (knots)  [primary]
  SPED → wind_speed     (knots, converted from m/s)  [fallback]
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WYOMING_BASE_URL = "http://weather.uwyo.edu/cgi-bin/sounding"
WYOMING_WSGI_URL = "http://weather.uwyo.edu/wsgi/sounding"

_MS_TO_KT = 1.94384  # m/s → knots conversion factor

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]


def _build_wyoming_url(station_id: str, sounding_dt: datetime) -> str:
    """
    Construct the University of Wyoming TEXT:LIST sounding URL.

    Parameters
    ----------
    station_id : str
        WMO station number (e.g. ``'16716'``).
    sounding_dt : datetime
        Sounding valid time.  Hour must be 0 or 12 (caller responsibility).

    Returns
    -------
    str
        Fully formed URL.
    """
    ddhh = f"{sounding_dt.day:02d}{sounding_dt.hour:02d}"
    return (
        f"{WYOMING_BASE_URL}"
        f"?region=europe"
        f"&TYPE=TEXT%3ALIST"
        f"&YEAR={sounding_dt.year}"
        f"&MONTH={sounding_dt.month:02d}"
        f"&FROM={ddhh}"
        f"&TO={ddhh}"
        f"&STNM={station_id}"
    )


def _build_wsgi_url(station_id: str, sounding_dt: datetime) -> str:
    """
    Construct the University of Wyoming WSGI/BUFR sounding URL (fallback).

    Parameters
    ----------
    station_id : str
        WMO station number (e.g. ``'08536'`` for Lisboa).
    sounding_dt : datetime
        Sounding valid time.  Hour must be 0 or 12 (caller responsibility).

    Returns
    -------
    str
        Fully formed URL.
    """
    dt_str = sounding_dt.strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"{WYOMING_WSGI_URL}"
        f"?datetime={quote(dt_str)}"
        f"&id={station_id}"
        f"&src=BUFR"
        f"&type=TEXT%3ALIST"
    )


def _parse_wyoming_text(response_text: str) -> pd.DataFrame:
    """
    Parse a Wyoming TEXT:LIST sounding response into a DataFrame.

    Parameters
    ----------
    response_text : str
        Full HTTP response body from the Wyoming server.

    Returns
    -------
    pd.DataFrame
        Columns: ``pressure`` (hPa), ``height`` (m), ``temperature`` (°C),
        ``dewpoint`` (°C), ``wind_direction`` (°), ``wind_speed`` (knots),
        ``u_wind`` (knots), ``v_wind`` (knots).

    Raises
    ------
    ValueError
        If the response contains ``"Can't get"`` or is empty after
        stripping, indicating no data is available.
    """
    text = response_text.strip()

    if not text:
        raise ValueError("Empty response from Wyoming server.")
    if "Can't get" in text:
        raise ValueError(
            "Wyoming server returned 'Can't get' — "
            "no data available for this station/time."
        )

    lines = text.splitlines()

    # Locate the header row (contains PRES, HGHT, TEMP, etc.)
    header_idx = None
    for i, line in enumerate(lines):
        if "PRES" in line and "HGHT" in line and "TEMP" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not locate column header row in response.")

    headers = lines[header_idx].split()

    # Data rows start after the dashed separator line
    data_start = None
    for i in range(header_idx + 1, len(lines)):
        if lines[i].strip().startswith("---"):
            data_start = i + 1
            break

    if data_start is None:
        data_start = header_idx + 1

    # Collect numeric data rows
    rows = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if line.startswith("<") or line.startswith("Station"):
            break
        parts = line.split()
        if len(parts) != len(headers):
            continue
        try:
            float(parts[0])
            rows.append(parts)
        except ValueError:
            continue

    if not rows:
        raise ValueError("No numeric data rows found in sounding response.")

    df_raw = pd.DataFrame(rows, columns=headers)

    col_map = {
        "PRES": "pressure",
        "HGHT": "height",
        "TEMP": "temperature",
        "DWPT": "dewpoint",
        "DRCT": "wind_direction",
    }

    df = pd.DataFrame()
    for raw_col, clean_col in col_map.items():
        if raw_col in df_raw.columns:
            df[clean_col] = pd.to_numeric(df_raw[raw_col], errors="coerce")
        else:
            df[clean_col] = np.nan

    # Wind speed: SKNT (knots) from cgi-bin; SPED (m/s) from wsgi — normalise
    # to knots so the rest of the pipeline always works in the same units.
    if "SKNT" in df_raw.columns:
        df["wind_speed"] = pd.to_numeric(df_raw["SKNT"], errors="coerce")
    elif "SPED" in df_raw.columns:
        df["wind_speed"] = pd.to_numeric(df_raw["SPED"], errors="coerce") * _MS_TO_KT
    else:
        df["wind_speed"] = np.nan

    df = df.dropna(subset=["pressure", "temperature"]).reset_index(drop=True)

    # Derive U/V wind components (knots)
    wdir_rad = np.radians(df["wind_direction"].values)
    wspd = df["wind_speed"].values
    df["u_wind"] = -wspd * np.sin(wdir_rad)
    df["v_wind"] = -wspd * np.cos(wdir_rad)

    return df


# [Core fetch]


def fetch_latest_sounding(
    station_id: str,
) -> tuple[pd.DataFrame, datetime]:
    """
    Fetch the most recently completed radiosonde sounding.

    Determines the latest completed sounding time based on
    ``datetime.now(timezone.utc)``:

    - UTC hour >= 6 and < 18  -> today 00 UTC
    - UTC hour >= 18          -> today 12 UTC
    - UTC hour < 6            -> yesterday 12 UTC

    Parameters
    ----------
    station_id : str
        WMO station number (e.g. ``'16622'`` for Thessaloniki).

    Returns
    -------
    tuple of (pd.DataFrame, datetime)
        ``(dataframe, sounding_datetime)`` — the DataFrame contains
        all parsed sounding levels; the datetime is the valid time.
    """
    now = datetime.now(timezone.utc)
    hour = now.hour

    if 6 <= hour < 18:
        sounding_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif hour >= 18:
        sounding_dt = now.replace(hour=12, minute=0, second=0, microsecond=0)
    else:  # hour < 6
        yesterday = now - timedelta(days=1)
        sounding_dt = yesterday.replace(
            hour=12, minute=0, second=0, microsecond=0
        )

    url = _build_wyoming_url(station_id, sounding_dt)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    try:
        df = _parse_wyoming_text(resp.text)
    except ValueError as exc:
        if "Can't get" not in str(exc):
            raise
        # Primary has no data — try the WSGI/BUFR fallback
        url = _build_wsgi_url(station_id, sounding_dt)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = _parse_wyoming_text(resp.text)

    if df.empty:
        raise ValueError(
            f"No sounding data returned for station {station_id} "
            f"at {sounding_dt.strftime('%Y-%m-%d %H:%M')} UTC."
        )

    return df, sounding_dt


def fetch_retrospective_sounding(
    station_id: str,
    date_str: str,
    hour: int,
) -> tuple[pd.DataFrame, datetime]:
    """
    Fetch a radiosonde sounding for a specific date and hour (0 or 12 UTC,
    since that's when most radiosondes are typically launched).

    Parameters
    ----------
    station_id : str
        WMO station id (e.g. ``'16622'``).
    date_str : str
        Date string in ``'YYYY-MM-DD'`` format.
    hour : int
        Sounding hour — must be ``0`` or ``12``.

    Returns
    -------
    tuple of (pd.DataFrame, datetime)
        ``(dataframe, sounding_datetime)`` — same contract as
        ``fetch_latest_sounding()``.
    """
    if hour not in (0, 12):
        raise ValueError(
            f"Sounding hour must be 0 or 12, got {hour}."
        )

    sounding_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hour, tzinfo=timezone.utc,
    )

    url = _build_wyoming_url(station_id, sounding_dt)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    try:
        df = _parse_wyoming_text(resp.text)
    except ValueError as exc:
        if "Can't get" not in str(exc):
            raise
        # Primary has no data — try the WSGI/BUFR fallback
        url = _build_wsgi_url(station_id, sounding_dt)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = _parse_wyoming_text(resp.text)

    if df.empty:
        raise ValueError(
            f"No sounding data returned for station {station_id} "
            f"at {sounding_dt.strftime('%Y-%m-%d %H:%M')} UTC."
        )

    return df, sounding_dt


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Example:")
    print("  from wyoming_raob import fetch_latest_sounding")
    print("  df, dt = fetch_latest_sounding('16622')")
