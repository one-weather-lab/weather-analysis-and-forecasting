#!/usr/bin/env python3
"""
Script Name: noaa_realtime.py
Purpose: Fetch the LATEST raw METAR for a set of ICAO stations
         directly from the NOAA NWS FTP server.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-02-28
Version: 2.0.0
License: MIT

Notes:
  • Inputs:  ICAO station code list. No date range — always fetches the
             single latest observation available on the server.
  • Outputs: pd.DataFrame with three columns — station (str), valid (datetime,
             UTC-naive), metar (str). Schema matches iem_raw.py output.
             Stations with no fresh data are silently omitted.
  • Configuration: MAX_AGE_MINUTES = 50 min (staleness cut-off); observations older
                   than this or more than 5 min in the future are discarded.
                   Per-station HTTP timeout is hardcoded to 10 s.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import certifi
import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NOAA_METAR_URL  = "https://tgftp.nws.noaa.gov/data/observations/metar/stations/{}.TXT"
MAX_AGE_MINUTES = 50
_MAX_AGE        = timedelta(minutes=MAX_AGE_MINUTES)
_FUTURE_TOL     = timedelta(minutes=5)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("owl.noaa.realtime")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]

def _parse_header_time(line: str) -> Optional[datetime]:
    """
    Parse the NOAA FTP file header timestamp (first line of .TXT file).

    The header uses one of two formats:
      YYYY/MM/DD HH:MM   or   YYYY-MM-DD HH:MM

    Parameters
    ----------
    line : str
        The first line of the NOAA FTP observation file.

    Returns
    -------
    datetime or None
        UTC-aware datetime if parsing succeeds, or None if parsing fails.
    """
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(line.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None

def _fetch_one(station: str, now_utc: datetime) -> Optional[dict]:
    """
    Fetch and validate the latest METAR for a single station.

    Observations are rejected when:
      - HTTP response is not 200
      - Header timestamp cannot be parsed
      - Age > MAX_AGE_MINUTES (stale)
      - Timestamp is more than 5 minutes in the future

    Parameters
    ----------
    station : str
        ICAO station code (e.g. 'LGIO').
    now_utc : datetime
        Current UTC-aware datetime used to compute observation age.

    Returns
    -------
    dict or None
        Dict with keys ``station``, ``valid``, ``metar`` if a fresh
        observation is available; None otherwise.
    """
    try:
        resp = requests.get(
            NOAA_METAR_URL.format(station),
            timeout=10,
            verify=certifi.where(),
        )
        if resp.status_code != 200:
            return None

        lines = [ln.strip() for ln in resp.text.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return None

        obs_time = _parse_header_time(lines[0])
        if obs_time is None:
            return None

        age = now_utc - obs_time
        if not (-_FUTURE_TOL <= age <= _MAX_AGE):
            return None

        return {
            "station": station,
            "valid"  : obs_time,
            "metar"  : lines[-1],   # last non-blank line is the METAR string
        }

    except Exception:
        return None

# [Core NOAA METAR fetcer]

def fetch_noaa_realtime(stations: List[str]) -> pd.DataFrame:
    """
    Fetch the latest raw METAR string for each station from NOAA NWS FTP.

    Only observations no older than ``MAX_AGE_MINUTES`` (default 50) are
    returned. 

    Parameters
    ----------
    stations : list of str
        ICAO station codes (e.g. ['LGIO', 'LGTS', 'LGAV']).

    Returns
    -------
    pd.DataFrame
        One row per station with columns: station, valid, metar.
        Stations with no fresh data are silently omitted.

    Example
    -------
    >>> df = fetch_noaa_realtime(['LGIO', 'LGTS'])
    >>> print(df[['station', 'metar']])
    """
    now_utc = datetime.now(timezone.utc)
    rows    = []
    skipped = []

    LOG.info("[NOAA] Fetching real-time METAR for %d station(s)...", len(stations))

    for stn in stations:
        result = _fetch_one(stn, now_utc)
        if result is not None:
            rows.append(result)
        else:
            skipped.append(stn)

    df = pd.DataFrame(rows, columns=["station", "valid", "metar"])

    if not df.empty:
        # tz_convert(None) strips the +00:00 suffix while keeping values as implicit UTC
        df["valid"] = pd.to_datetime(df["valid"], utc=True).dt.tz_convert(None)

    LOG.info("[NOAA] Retrieved %d observation(s).", len(df))
    if skipped:
        LOG.info("[NOAA] Skipped %d station(s) (no data <= %d min): %s",
                 len(skipped), MAX_AGE_MINUTES, ", ".join(skipped))

    return df

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Example:")
    print("  from noaa_realtime import fetch_noaa_realtime")
    print("  df = fetch_noaa_realtime(['LGIO', 'LGTS'])")
