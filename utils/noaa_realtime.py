#!/usr/bin/env python3
"""
Script Name: noaa_realtime.py
Purpose: Fetch the LATEST raw METAR for a set of ICAO stations
         directly from the NOAA NWS FTP server.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-04-13
Version: 3.0.0
License: MIT

Notes:
  • Inputs:  ICAO station code list. No date range — always fetches the
             single latest observation available on the server.
  • Outputs: pd.DataFrame with three columns — station (str), valid (datetime,
             UTC-naive), metar (str). Schema matches iem_raw.py output.
             Stations with no fresh data are silently omitted.
  • Parallelism: stations are fetched concurrently via ThreadPoolExecutor
                 (MAX_WORKERS threads for the initial round, RETRY_WORKERS
                 for retry rounds).
  • Configuration: MAX_AGE_MINUTES = 50 min (staleness cut-off); observations older
                   than this or more than 5 min in the future are discarded.
                   MAX_WORKERS = 32, RETRY_WORKERS = 8, N_RETRIES = 3,
                   RETRY_BASE_S = 3.0 s (exponential back-off base).
                   Retryable errors: timeout, SSL, connection error, HTTP 429/5xx.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import certifi
import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NOAA_METAR_URL = "https://tgftp.nws.noaa.gov/data/observations/metar/stations/{}.TXT"
MAX_AGE_MINUTES = 50      # staleness cut-off (minutes)
MAX_WORKERS     = 32      # threads for initial parallel fetch
RETRY_WORKERS   = 8       # threads for retry rounds (lower = gentler on server)
N_RETRIES       = 3       # max retry attempts for transient failures
RETRY_BASE_S    = 3.0     # back-off base delay; actual delay = RETRY_BASE_S * 2^round

_MAX_AGE    = timedelta(minutes=MAX_AGE_MINUTES)
_FUTURE_TOL = timedelta(minutes=5)

# HTTP status codes that are worth retrying (server-side / rate-limit transients)
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}

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


def _is_retryable(reason: str) -> bool:
    """
    Return True when a fetch failure is transient and worth retrying.

    Retryable
    ---------
    - ``exception(*)``         : timeout, SSL error, connection reset, …
    - ``http_429``             : rate-limited by server
    - ``http_5xx``             : server-side error (500/502/503/504)

    Not retryable
    -------------
    - ``http_404``             : no file on NOAA (station simply does not report there)
    - ``http_4xx`` (other)     : client-side error unlikely to resolve on retry
    - ``stale`` / ``future``   : data-quality filter — retry cannot make old data fresh
    - ``parse_error``          : malformed response unlikely to change on retry
    """
    if reason.startswith("exception("):
        return True
    if reason.startswith("http_"):
        try:
            code = int(reason.split("_", 1)[1])
            return code in _RETRYABLE_HTTP
        except (ValueError, IndexError):
            return False
    return False


def _fetch_one(station: str, now_utc: datetime) -> Tuple[Optional[dict], Optional[str]]:
    """
    Fetch and validate the latest METAR for a single station.

    Parameters
    ----------
    station : str
        ICAO station code (e.g. 'LGIO').
    now_utc : datetime
        Current UTC-aware datetime used to compute observation age.

    Returns
    -------
    tuple (dict | None, str | None)
        ``(record, None)``   — success; record has keys station/valid/metar.
        ``(None, reason)``   — failure; reason string encodes the cause:

        ================== =====================================================
        ``http_<code>``    HTTP status code was not 200 (e.g. ``http_404``)
        ``parse_error``    Response body could not be parsed
        ``stale(<N>min)``  Observation is older than MAX_AGE_MINUTES
        ``future(<N>min)`` Timestamp is more than 5 min ahead of now_utc
        ``exception(<T>)`` Network or SSL exception of type T
        ================== =====================================================
    """
    try:
        resp = requests.get(
            NOAA_METAR_URL.format(station),
            timeout=10,
            verify=certifi.where(),
        )
        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"

        lines = [ln.strip() for ln in resp.text.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return None, "parse_error"

        obs_time = _parse_header_time(lines[0])
        if obs_time is None:
            return None, "parse_error"

        age     = now_utc - obs_time
        age_min = age.total_seconds() / 60
        if age < -_FUTURE_TOL:
            return None, f"future({age_min:.0f}min)"
        if age > _MAX_AGE:
            return None, f"stale({age_min:.0f}min)"

        return {
            "station": station,
            "valid"  : obs_time,
            "metar"  : lines[-1],   # last non-blank line is the METAR string
        }, None

    except Exception as exc:
        exc_name = type(exc).__name__
        return None, f"exception({exc_name})"


def _fetch_batch(
    stations: List[str],
    now_utc: datetime,
    max_workers: int,
) -> Tuple[List[dict], List[str], Dict[str, int]]:
    """
    Fetch stations in parallel and partition results.

    Parameters
    ----------
    stations : list of str
        ICAO station codes (e.g. 'LGIO').
    now_utc : datetime
        Current UTC-aware datetime used to compute observation age.
    max_workers : int
        Number of parallel threads to use for the fetch.

    Returns
    -------
    rows : list of dict
        Successfully retrieved records.
    to_retry : list of str
        Station codes that failed with a transient (retryable) error.
    perm_counts : dict
        Permanent skip reason → count  (for logging).
    """
    rows       : List[dict]      = []
    to_retry   : List[str]       = []
    perm_counts: Dict[str, int]  = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_stn = {pool.submit(_fetch_one, stn, now_utc): stn
                         for stn in stations}
        for future in as_completed(future_to_stn):
            stn            = future_to_stn[future]
            record, reason = future.result()
            if record is not None:
                rows.append(record)
            elif _is_retryable(reason):
                to_retry.append(stn)
            else:
                bucket = reason.split("(")[0]   # e.g. 'stale(62min)' -> 'stale'
                perm_counts[bucket] = perm_counts.get(bucket, 0) + 1

    return rows, to_retry, perm_counts

# [Core NOAA METAR fetcher]

def fetch_noaa_realtime(stations: List[str]) -> pd.DataFrame:
    """
    Fetch the latest raw METAR string for each station from NOAA NWS FTP.

    Only observations no older than ``MAX_AGE_MINUTES`` (default 50) are
    returned. Retries are performed for transient errors.

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

    LOG.info(
        "[NOAA] Fetching METAR for %d station(s)  |  workers=%d  |  cut-off=%d min  |  "
        "retries=%d  |  ref UTC %s",
        len(stations), MAX_WORKERS, MAX_AGE_MINUTES, N_RETRIES,
        now_utc.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # ── Round 0: initial full fetch ───────────────────────────────────────────
    all_rows, to_retry, perm_counts = _fetch_batch(stations, now_utc, MAX_WORKERS)

    LOG.info(
        "[NOAA] Round 0: %d ok  |  %d retryable  |  %d permanent-skip",
        len(all_rows), len(to_retry), sum(perm_counts.values()),
    )

    # ── Retry rounds for transient failures ───────────────────────────────────
    for attempt in range(1, N_RETRIES + 1):
        if not to_retry:
            break

        delay = RETRY_BASE_S * (2 ** (attempt - 1))   # 3 s, 6 s, 12 s
        LOG.info(
            "[NOAA] Retry %d/%d: %d station(s) pending  |  back-off %.0f s  "
            "|  workers=%d",
            attempt, N_RETRIES, len(to_retry), delay, RETRY_WORKERS,
        )
        time.sleep(delay)

        rows, to_retry, round_perm = _fetch_batch(to_retry, now_utc, RETRY_WORKERS)
        all_rows.extend(rows)

        # Accumulate permanent skips from this round
        for k, v in round_perm.items():
            perm_counts[k] = perm_counts.get(k, 0) + v

        LOG.info(
            "[NOAA] Retry %d/%d result: +%d ok  |  %d still retryable",
            attempt, N_RETRIES, len(rows), len(to_retry),
        )

    # ── Final accounting ──────────────────────────────────────────────────────
    # Stations still in to_retry exhausted all attempts — treat as transient skip
    if to_retry:
        perm_counts["exhausted_retries"] = len(to_retry)
        LOG.warning(
            "[NOAA] %d station(s) never responded after %d retries: %s",
            len(to_retry), N_RETRIES, ", ".join(sorted(to_retry)),
        )

    df = pd.DataFrame(all_rows, columns=["station", "valid", "metar"])
    if not df.empty:
        df["valid"] = pd.to_datetime(df["valid"], utc=True).dt.tz_convert(None)

    total_skip = sum(perm_counts.values())
    LOG.info("[NOAA] Retrieved %d observation(s).", len(df))
    if perm_counts:
        summary = "  |  ".join(f"{k}: {v}" for k, v in sorted(perm_counts.items()))
        LOG.info("[NOAA] Permanent skips: %d  —  %s", total_skip, summary)

    return df

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Example:")
    print("  from noaa_realtime import fetch_noaa_realtime")
    print("  df = fetch_noaa_realtime(['LGIO', 'LGTS'])")
