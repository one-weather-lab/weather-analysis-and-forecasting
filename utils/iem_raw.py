#!/usr/bin/env python3
"""
Script Name: iem_raw.py
Purpose: Retrieve raw METAR from the Iowa Environmental Mesonet (IEM) ASOS service.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-04-13
Version: 3.1.0
License: MIT

Notes:
  • Context: Uses data=metar (raw-string mode), NOT data=all (decoded CSV),
             so that decoding is handled externally by metar_helpers.py.
  • Inputs:  ICAO station code list; date range as YYYY-MM-DD strings (UTC).
             URL parameters: format=onlycomma, tz=UTC, report_type=3 (routine).
  • Outputs: pd.DataFrame with three columns — station (str), valid (datetime,
             UTC-naive), metar (str). Schema matches noaa_realtime.py output.
  • Configuration: Timeout 120 s. Station list batched at _STATION_BATCH_SIZE = 20
                   to avoid HTTP 414 (Request-URI Too Long) on large networks.
                   Missing values encoded as 'M' by IEM are dropped.
                   Trace precipitation is passed through as 'T'.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import io
import logging
from datetime import datetime
from typing import List

import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
_IEM_BASE_URL = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
_REQUEST_TIMEOUT_S = 120          # seconds; increase for large multi-station requests
_STATION_BATCH_SIZE = 20          # max stations per request (avoids HTTP 414)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("owl.iem.raw")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]


def _build_url(stations: List[str], start_date: str, end_date: str) -> str:
    """
    Build IEM ASOS request URL for multiple stations and a date range.

    Parameters
    ----------
    stations : list of str
        ICAO station codes.
    start_date, end_date : str
        Date strings in YYYY-MM-DD format (inclusive, UTC).

    Returns
    -------
    str
        Complete IEM ASOS API URL returning raw METAR strings.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    stn_params = "&".join(f"station={s}" for s in stations)

    return (
        f"{_IEM_BASE_URL}{stn_params}"
        "&data=metar"                              # ← raw METAR strings only
        f"&year1={start.year}&month1={start.month}&day1={start.day}"
        f"&year2={end.year}&month2={end.month}&day2={end.day}"
        "&tz=Etc%2FUTC"
        "&format=onlycomma"
        "&missing=M"
        "&trace=T"
        "&report_type=3"                           # routine observations
    )

# [Core IEM METAR fetcher]

def fetch_iem_raw(stations: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch raw METAR telegram strings from the IEM ASOS archive.

    Parameters
    ----------
    stations : list of str
        ICAO station codes (e.g. ['LGIO', 'LGTS']).
    start_date : str
        Start date in YYYY-MM-DD format (UTC).
    end_date : str
        End date in YYYY-MM-DD format (UTC, inclusive).

    Returns
    -------
    pd.DataFrame
        Three columns:
          - ``station`` : ICAO code (str)
          - ``valid``   : observation timestamp (datetime, UTC-naive, implicit UTC)
          - ``metar``   : raw METAR telegram string (str)

        Rows with missing or unparseable timestamps are dropped.
        Sorted by station and time.

    Example
    -------
    >>> df = fetch_iem_raw(['LGIO'], '2021-08-03', '2021-08-03')
    >>> print(df[['station', 'metar']].head(3))
    """
    LOG.info("[IEM] Fetching raw METAR for %d station(s)  %s -> %s",
             len(stations), start_date, end_date)

    # Split into batches to avoid HTTP 414 (Request-URI Too Long)
    batches = [
        stations[i:i + _STATION_BATCH_SIZE]
        for i in range(0, len(stations), _STATION_BATCH_SIZE)
    ]
    if len(batches) > 1:
        LOG.info("[IEM] Splitting into %d batches of <= %d stations",
                 len(batches), _STATION_BATCH_SIZE)

    frames = []
    for batch_idx, batch in enumerate(batches):
        url = _build_url(batch, start_date, end_date)
        try:
            r = requests.get(url, timeout=_REQUEST_TIMEOUT_S)
            r.raise_for_status()
        except requests.exceptions.RequestException as exc:
            LOG.error("[IEM] Batch %d/%d failed: %s", batch_idx + 1, len(batches), exc)
            raise

        batch_df = pd.read_csv(io.StringIO(r.text))
        if not batch_df.empty:
            frames.append(batch_df)

    if not frames:
        LOG.warning("[IEM] No data returned.")
        return pd.DataFrame(columns=["station", "valid", "metar"])

    df = pd.concat(frames, ignore_index=True)

    # Ensure the three expected columns exist
    for col in ("station", "valid", "metar"):
        if col not in df.columns:
            df[col] = pd.NA

    # Parse timestamps (IEM uses UTC, format: "YYYY-MM-DD HH:MM")
    # tz_convert(None) strips the +00:00 suffix while keeping values as implicit UTC
    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["valid"])

    # Drop rows with no METAR string
    df = df[df["metar"].notna() & (df["metar"].str.strip() != "M")]

    # Keep only the three core columns
    df = df[["station", "valid", "metar"]].copy()

    df = df.sort_values(["station", "valid"]).reset_index(drop=True)

    LOG.info("[IEM] Retrieved %d observation(s) from %d station(s).",
             len(df), df["station"].nunique())
    return df

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Example:")
    print("  from iem_raw import fetch_iem_raw")
    print("  df = fetch_iem_raw(['LGIO'], '2021-08-03', '2021-08-03')")
