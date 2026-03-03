#!/usr/bin/env python3
"""
Script Name: metar_helpers.py
Purpose: Parse raw FM-15 METAR strings and provide unit-conversion and QC helpers.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-02-28
Version: 4.0.0
License: MIT

Notes:
  • Inputs:  Raw FM-15 METAR alphanumeric strings (WMO/ICAO International and US (FAA) formats).
  • Outputs: Dictionary of decoded meteorological fields per observation (see schema
             below); augmented pd.Series rows via decode_metar_row().
  • Configuration: Unit-conversion factors and QC thresholds are defined as
                   module-level constants in the Configuration section.

  Column schema produced by parse_metar_string()
  -----------------------------------------------
  wdir             : wind direction (degrees; nan = variable/VRB)
  wspd             : wind speed (knots, native METAR unit)
  wspd_ms          : wind speed (m/s, SI)
  gust_ms          : gust speed (m/s; nan = no gust reported)
  wdir_from        : variable wind range lower bound (degrees; nan if not reported)
  wdir_to          : variable wind range upper bound (degrees; nan if not reported)
  temp_c           : air temperature (degC)
  dwpt_c           : dewpoint temperature (degC)
  relh             : relative humidity (%, derived via Magnus formula)
  mslp             : mean sea-level pressure (hPa)
  vsby_m           : prevailing visibility (metres)
  cloud_cover      : dominant sky cover in oktas (0-8, integer; WMO SYNOP code)
  cloud_base_m     : height of dominant cloud layer (metres)
  skyc1-4          : sky cover code per slot in the report (CLR/FEW/SCT/BKN/OVC/VV/CAVOK)
  skyl1-4          : cloud base height per slot in the report (metres)
  cloud_cover_low  : oktas of dominant low cloud layer (base < 2000 m; None if absent)
  cloud_base_low_m : base height of dominant low cloud layer (metres)
  cloud_cover_mid  : oktas of dominant mid cloud layer (2000-5999 m; None if absent)
  cloud_base_mid_m : base height of dominant mid cloud layer (metres)
  cloud_cover_high : oktas of dominant high cloud layer (base >= 6000 m; None if absent)
  cloud_base_high_m: base height of dominant high cloud layer (metres)
  wxcodes          : significant weather string (e.g. '-RA BR')
  u_kt, v_kt       : MetPy barb components (knots; for Section 3 plots only)
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import math
import re
from typing import Dict, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# [Compiled FM-15 METAR regex patterns]

# Wind: 25010KT, 25010G20KT, VRB05KT, 00000KT, 25010MPS, 25010G20MPS, 25010KMH
_RX_WIND = re.compile(
    r"\b(?P<dir>\d{3}|VRB)"
    r"(?P<spd>\d{2,3})"
    r"(?:G(?P<gst>\d{2,3}))?"
    r"(?P<unit>KT|KMH|MPS)\b"
)

# Variable wind direction range, e.g. 270V360
_RX_WIND_VAR = re.compile(r"\b(\d{3})V(\d{3})\b")

# Sky layer: FEW018, SCT040CB, BKN100TCU, OVC010, VV002
_RX_SKY_LAYER = re.compile(
    r"\b(?P<code>FEW|SCT|BKN|OVC|VV)(?P<hgt>\d{3})(?:CB|TCU)?\b"
)

# Clear-sky codes (no height)
_RX_SKY_CLEAR = re.compile(r"\b(?P<code>CLR|SKC|NSC|NCD)\b")

# Temperature / Dewpoint: 10/05, 04/M04, M03/M08
# Requires exactly 2 chars per field (with optional M prefix)
_RX_T_TD = re.compile(r"(?<!\w)(?P<T>M?\d{2})/(?P<Td>M?\d{2})(?!\d)")

# QNH: Q1013 (hPa, WMO/ICAO) or A2992 (inHg x100, US altimeter)
_RX_QNH  = re.compile(r"\bQ(?P<hpa>\d{4})\b")
_RX_ALTI = re.compile(r"\bA(?P<inhg>\d{4})\b")

# RMK T-group — 0.1 degC precision: T00391044 -> T+3.9 C, Td-4.4 C
_RX_T_GROUP = re.compile(r"\bT(?P<ts>[01]\d{3})(?P<tds>[01]\d{3})\b")

# RMK MSLP: MSLP022 -> 1002.2 hPa
_RX_MSLP = re.compile(r"\bMSLP(?P<mslp>\d{3})\b")

# Runway Visual Range: R09/0600U, R28L/P1200
# Note: RVR tokens are implicitly skipped by the visibility parser (no standalone
# 4-digit match on R##/... tokens). Explicit parsing of RVR values is not
# implemented; _RX_RVR is retained here as a pattern reference only.
_RX_RVR = re.compile(r"\bR\d{2}[LCR]?/[MP]?\d{4}[UDN]?\b")

# Significant weather (ICAO)
_RX_WX = re.compile(
    r"\b[+-]?"
    r"(?:MI|BC|PR|DR|BL|SH|TS|FZ)?"
    r"(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP)+"
    r"\b"
    r"|"
    r"\b(?:BR|FG|FU|VA|DU|SA|HZ|PO|SQ|FC|SS|DS)\b"
)

# Visibility: ICAO 4-digit meters (standalone)
_RX_VIS_M = re.compile(r"(?<!\S)(\d{4})(?!\S)")

# Visibility: US statute miles (e.g. 10SM, 1 1/2SM, 3/4SM)
_RX_VIS_SM = re.compile(r"(?P<vis>(?:\d+\s+)?\d+/\d+|\d+)\s*SM\b")

# Less-than visibility (M1/4SM)
_RX_VIS_SM_LT = re.compile(r"M(?P<vis>\d+/\d+)\s*SM\b")

# METAR/SPECI, COR, AUTO header tokens to strip
_RX_HEADER = re.compile(
    r"^(?:(?:METAR|SPECI)\s+)?(?:COR\s+)?[A-Z]{4}\s+\d{6}Z\s+(?:AUTO\s+|COR\s+)?"
)

# [Constants]

# Trend/supplementary keywords that terminate the body
_TREND_KEYWORDS = (" NOSIG", " BECMG", " TEMPO", " PROB", " INTER")

# Feet to metres conversion
_FT_TO_M = 0.3048

# Sky cover ranking (for selecting the dominant layer per observation)
_COVER_ORDER = {
    "CLR": 0, "SKC": 0, "NSC": 0, "NCD": 0, "CAVOK": 0,
    "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 4,
}

# Sky cover in WMO SYNOP oktas (0-8):
#   CLR/SKC/NSC  = 0   (no cloud)
#   FEW (1-2/8)  = 2   (use upper bound of range)
#   SCT (3-4/8)  = 4   (use upper bound of range)
#   BKN (5-7/8)  = 6   (use midpoint of range)
#   OVC (8/8)    = 8   (overcast)
#   VV           = 8   (vertical visibility; sky obscured)
_COVER_OKTAS = {
    "CLR": 0, "SKC": 0, "NSC": 0, "NCD": 0, "CAVOK": 0,
    "FEW": 2, "SCT": 4, "BKN": 6, "OVC": 8, "VV": 8,
}

# Oktas lookup for sky_cover_from_code() (Section 3 station model plots)
CLOUD_COVER_MAP = {
    "SKC": 0, "CLR": 0, "NSC": 0, "NCD": 0, "CAVOK": 0,
    "FEW": 2, "SCT": 4, "BKN": 6, "OVC": 8, "VV": 8,
}


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]

def _m_to_celsius(token: str) -> float:
    """
    Convert a METAR temperature token to degrees Celsius.

    The METAR convention uses 'M' as a prefix for negative temperatures
    (e.g. 'M03' = -3 degC).

    Parameters
    ----------
    token : str
        Temperature token from a decoded METAR string (e.g. '05', 'M03').

    Returns
    -------
    float
        Temperature in degrees Celsius.
    """
    token = str(token).strip().upper()
    return -float(token[1:]) if token.startswith("M") else float(token)


def _decode_t_group(ts: str, tds: str) -> Tuple[float, float]:
    """
    Decode RMK T-group to (temp_c, dwpt_c) with 0.1 degC precision.

    Format: T<s><3digits><s><3digits>  (s: 0=positive, 1=negative)
    e.g. T00391044 -> T = +3.9 C, Td = -4.4 C

    Parameters
    ----------
    ts : str
        4-character temperature substring (sign digit + 3 value digits).
    tds : str
        4-character dewpoint substring (sign digit + 3 value digits).

    Returns
    -------
    tuple of (float, float)
        (temperature in degC, dewpoint in degC) with 0.1 degC precision.
    """
    sign_t  = -1 if ts[0]  == "1" else 1
    sign_td = -1 if tds[0] == "1" else 1
    return sign_t * int(ts[1:]) / 10.0, sign_td * int(tds[1:]) / 10.0


def _decode_mslp(mslp_str: str) -> float:
    """
    Decode RMK MSLP group to mean sea-level pressure in hPa.

    If the 3-digit value >= 550, the leading digit is '9' (i.e. 9xx.x hPa);
    otherwise it is '10' (i.e. 10xx.x hPa).

    Parameters
    ----------
    mslp_str : str
        3-digit MSLP string from the RMK section (e.g. '022', '992').

    Returns
    -------
    float
        Mean sea-level pressure in hPa (e.g. 1002.2, 999.2).
    """
    val = int(mslp_str)
    return (900 + val / 10.0) if val >= 550 else (1000 + val / 10.0)


def _vis_sm_to_m(vis_str: str) -> float:
    """
    Parse a US visibility string in statute miles to metres.

    Handles whole numbers ('10'), fractions ('3/4'), and mixed forms
    ('1 1/2').

    Parameters
    ----------
    vis_str : str
        Visibility string in statute miles.

    Returns
    -------
    float
        Visibility in metres (1 SM = 1609.344 m).
    """
    vis_str = vis_str.strip()
    if " " in vis_str:
        whole, frac = vis_str.split()
        num, den = frac.split("/")
        sm = float(whole) + float(num) / float(den)
    elif "/" in vis_str:
        num, den = vis_str.split("/")
        sm = float(num) / float(den)
    else:
        sm = float(vis_str)
    return sm * 1609.344


def _magnus_rh(temp_c: float, dwpt_c: float) -> float:
    """
    Compute relative humidity (%) from temperature and dewpoint.

    Uses the Magnus approximation with coefficients a = 17.625 and
    b = 243.04 degC.

    Parameters
    ----------
    temp_c : float
        Air temperature in degrees Celsius.
    dwpt_c : float
        Dewpoint temperature in degrees Celsius.

    Returns
    -------
    float
        Relative humidity in percent [0, 100], rounded to 1 decimal.

    Reference(s)
    ----------
    Alduchov, O.A. & Eskridge, R.E. (1996).
    Improved Magnus form approximation of saturation vapor pressure.
    Journal of Applied Meteorology and Climatology, 35(4), 601-609.
    https://doi.org/10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2
    """
    return round(
        100.0 * math.exp(17.625 * dwpt_c / (243.04 + dwpt_c))
               / math.exp(17.625 * temp_c / (243.04 + temp_c)),
        1,
    )

# [Core FM-15 METAR parser]

def parse_metar_string(metar_str: str) -> Dict:
    """
    Parse a raw FM-15 METAR string into SI-consistent meteorological variables.

    Handles the full FM-15 format:

        [METAR|SPECI] [COR] ICAO DDHHMMZ [AUTO]
        wind [direction-variability]
        [visibility] [min-vis] [RVR]
        [weather]
        sky-conditions
        T/Td
        [Q|A]pressure
        [NOSIG|BECMG|TEMPO|PROB...]
        [RMK ...]

    Supports:
      - WMO/ICAO: temperature in degC, Q group (hPa), vis in metres
      - US (FAA): A group (inHg), vis in SM, RMK T-group for 0.1-degC precision
      - Sub-zero temperature (M prefix, e.g. M03/M05)
      - Variable wind direction (e.g. 270V360)
      - CAVOK (visibility >= 10 km, no cloud below 5000 ft, no significant wx)
      - Missing AUTO-station data (e.g. //// for visibility, ////// for sky)

    Parameters
    ----------
    metar_str : str
        Raw METAR telegram string.

    Returns
    -------
    dict
        Keys:
          wdir, wspd, wspd_ms, gust_ms, wdir_from, wdir_to,
          temp_c, dwpt_c, relh, mslp, vsby_m,
          cloud_cover, cloud_base_m,
          skyc1-4, skyl1-4 (metres),
          wxcodes, u_kt, v_kt

    Reference(s)
    ----------
    World Meteorological Organization (2019).
    Manual on Codes — International Codes, Volume I.1:
    Annex II to the WMO Technical Regulations, Part A — Alphanumeric Codes,
    FM 15–XV METAR / FM 16–XV SPECI.
    WMO-No. 306, 2019 edition.
    https://library.wmo.int/idurl/4/35713

    Stull, R. B. (2018).
    Practical Meteorology: An Algebra-based Survey of Atmospheric Science, -version 1.02 edn.
    University of British Columbia, Vancouver, 940 pages.  isbn 978-0-88865-283-6
    """
    result: Dict = {}

    if pd.isna(metar_str) or not str(metar_str).strip():
        return result

    metar_str = str(metar_str).strip()

    # ── Split RMK section ─────────────────────────────────────────────────────
    rmk_idx = metar_str.find(" RMK ")
    body = metar_str[:rmk_idx] if rmk_idx >= 0 else metar_str
    rmk  = metar_str[rmk_idx + 5:] if rmk_idx >= 0 else ""

    # ── Truncate at trend indicators (NOSIG, BECMG, TEMPO, PROB, INTER) ──────
    for kw in _TREND_KEYWORDS:
        idx = body.find(kw)
        if idx >= 0:
            body = body[:idx]

    # ── Strip preamble: METAR/SPECI, COR, station ID, time, AUTO/COR ─────────
    m_hdr = _RX_HEADER.match(body)
    body_stripped = body[m_hdr.end():].strip() if m_hdr else body

    # ── Wind ──────────────────────────────────────────────────────────────────
    m_wind = _RX_WIND.search(body_stripped)
    if m_wind:
        d    = m_wind.group("dir")
        spd  = float(m_wind.group("spd"))
        gst  = m_wind.group("gst")
        unit = m_wind.group("unit")

        result["wdir"] = float("nan") if d == "VRB" else float(d)

        if unit == "MPS":
            spd_kt = spd * 1.94384
        elif unit == "KMH":
            spd_kt = spd / 1.852
        else:  # KT
            spd_kt = spd
        result["wspd"]    = spd_kt                    # knots (native METAR unit)
        result["wspd_ms"] = knots_to_ms(spd_kt)       # m/s (SI)

        if gst:
            if unit == "MPS":
                gst_kt = float(gst) * 1.94384
            elif unit == "KMH":
                gst_kt = float(gst) / 1.852
            else:  # KT
                gst_kt = float(gst)
            result["gust_ms"] = knots_to_ms(gst_kt)
        else:
            result["gust_ms"] = float("nan")

        # U/V barb components (internal, in knots for MetPy)
        if not math.isnan(result["wdir"]):
            result["u_kt"], result["v_kt"] = knots_to_uv_barbs(spd_kt, result["wdir"])
        else:
            result["u_kt"] = result["v_kt"] = float("nan")

    # ── Variable wind direction range (e.g. 270V360) ─────────────────────────
    m_var = _RX_WIND_VAR.search(body_stripped)
    if m_var:
        result["wdir_from"] = float(m_var.group(1))
        result["wdir_to"]   = float(m_var.group(2))
    else:
        result["wdir_from"] = result["wdir_to"] = float("nan")

    # ── Visibility ────────────────────────────────────────────────────────────
    if "CAVOK" in body_stripped:
        result["vsby_m"] = 10000.0          # CAVOK implies >= 10 km vis
    elif _RX_VIS_SM_LT.search(body_stripped):
        result["vsby_m"] = 0.0              # M1/4SM = less than 1/4 SM
    else:
        m_sm = _RX_VIS_SM.search(body_stripped)
        if m_sm:
            result["vsby_m"] = _vis_sm_to_m(m_sm.group("vis"))
        else:
            # ICAO 4-digit meters: find the first standalone 4-digit token
            # after the wind group; skip if it looks like a time or height
            wind_end_pos = m_wind.end() if m_wind else 0
            for token in body_stripped[wind_end_pos:].split():
                if re.fullmatch(r"\d{4}", token):
                    raw_vis = float(token)
                    # ICAO codes 9999 as ">= 10 km"
                    result["vsby_m"] = 10000.0 if raw_vis == 9999.0 else raw_vis
                    break

    # ── Sky conditions (up to 4 slots in the report, heights in METRES) ────────────────────
    if "CAVOK" in body_stripped:
        result.update({
            "cloud_cover": 0,    "cloud_base_m": float("nan"),
            "skyc1": "CAVOK",    "skyl1": None,
            "skyc2": None,       "skyl2": None,
            "skyc3": None,       "skyl3": None,
            "skyc4": None,       "skyl4": None,
            # All cloud levels clear under CAVOK
            "cloud_cover_low":  0,             "cloud_base_low_m":  float("nan"),
            "cloud_cover_mid":  0,             "cloud_base_mid_m":  float("nan"),
            "cloud_cover_high": 0,             "cloud_base_high_m": float("nan"),
        })
    else:
        slots = _RX_SKY_LAYER.findall(body_stripped)   # [(code, hgt_str), ...]
        m_clear = _RX_SKY_CLEAR.search(body_stripped)

        best_rank   = -1
        cloud_cover = -1        # -1 = not yet determined (nan equivalent for oktas)
        cloud_base  = float("nan")

        for n in range(1, 5):
            if n - 1 < len(slots):
                code, hgt_str = slots[n - 1]
                hgt_m = float(hgt_str) * 100.0 * _FT_TO_M   # hundreds of ft -> m
                result[f"skyc{n}"] = code
                result[f"skyl{n}"] = hgt_m
                rank = _COVER_ORDER.get(code, -1)
                if rank > best_rank:
                    best_rank   = rank
                    cloud_cover = _COVER_OKTAS.get(code, -1)
                    cloud_base  = hgt_m
            else:
                if m_clear and n == 1:
                    result["skyc1"] = m_clear.group("code")
                    result["skyl1"] = None
                    cloud_cover = 0
                result.setdefault(f"skyc{n}", None)
                result.setdefault(f"skyl{n}", None)

        result["cloud_cover"]  = cloud_cover
        result["cloud_base_m"] = cloud_base

        # ── Per-level cloud classification ────────────────────────────────────
        # WMO approximate height thresholds (mid-latitude, metres):
        #   Low  : base <  2000 m  (St, Sc, Ns, Cu)
        #   Mid  : base 2000–5999 m  (As, Ac)
        #   High : base >= 6000 m  (Ci, Cc, Cs)
        _LEVEL_BOUNDS = [
            ("low",  0.0,    2000.0),
            ("mid",  2000.0, 6000.0),
            ("high", 6000.0, float("inf")),
        ]
        for _lev, _lo, _hi in _LEVEL_BOUNDS:
            _best_ok   = -1
            _best_base = float("nan")
            for n in range(1, 5):
                code = result.get(f"skyc{n}")
                hgt  = result.get(f"skyl{n}")
                if code is None or hgt is None or math.isnan(hgt):
                    continue
                if _lo <= hgt < _hi:
                    oktas = _COVER_OKTAS.get(code, -1)
                    if oktas > _best_ok:
                        _best_ok   = oktas
                        _best_base = hgt
            result[f"cloud_cover_{_lev}"] = _best_ok if _best_ok >= 0 else None
            result[f"cloud_base_{_lev}_m"] = _best_base

    # ── Temperature / Dewpoint ────────────────────────────────────────────────
    # Prefer RMK T-group (0.1 degC precision, US stations)
    if rmk:
        m = _RX_T_GROUP.search(rmk)
        if m:
            result["temp_c"], result["dwpt_c"] = _decode_t_group(
                m.group("ts"), m.group("tds")
            )

    if "temp_c" not in result:
        m = _RX_T_TD.search(body_stripped)
        if m:
            result["temp_c"] = _m_to_celsius(m.group("T"))
            result["dwpt_c"] = _m_to_celsius(m.group("Td"))

    # ── Derived relative humidity (Magnus formula) ────────────────────────────
    T, Td = result.get("temp_c"), result.get("dwpt_c")
    if T is not None and Td is not None and not (math.isnan(T) or math.isnan(Td)):
        result["relh"] = _magnus_rh(T, Td)

    # ── Sea-level pressure (mslp) ─────────────────────────────────────────────
    m_q = _RX_QNH.search(body_stripped)
    m_a = _RX_ALTI.search(body_stripped)
    if m_q:
        result["mslp"] = float(m_q.group("hpa"))
    elif m_a:
        # A-group: 4-digit inHg × 100  (e.g. A2992 = 29.92 inHg)
        result["mslp"] = inhg_to_hpa(float(m_a.group("inhg")) / 100.0)

    # SLP from remarks (fallback, when no Q/A group)
    if "mslp" not in result and rmk:
        m = _RX_SLP.search(rmk)
        if m:
            result["mslp"] = _decode_slp(m.group("mslp"))

    # ── Significant weather ───────────────────────────────────────────────────
    wx = _RX_WX.findall(body_stripped)
    result["wxcodes"] = " ".join(w.strip() for w in wx if w.strip())

    return result


# [DataFrame-level decoder]

def decode_metar_row(row: pd.Series) -> pd.Series:
    """
    Decode a single raw METAR row by calling parse_metar_string() on the
    ``metar`` column.  Works identically for IEM and NOAA real-time rows.

    Parameters
    ----------
    row : pd.Series
        A single row from a DataFrame containing at least a ``metar`` column
        with the raw METAR telegram string.

    Returns
    -------
    pd.Series
        The input row augmented with all decoded meteorological fields
        produced by ``parse_metar_string()``.
    """
    row = row.copy()
    decoded = parse_metar_string(row.get("metar"))
    for key, val in decoded.items():
        row[key] = val
    return row


# [Unit conversion helpers]

def knots_to_ms(speed_kt: float) -> float:
    """
    Convert wind speed from knots to metres per second.

    Parameters
    ----------
    speed_kt : float
        Wind speed in knots. NaN values are passed through.

    Returns
    -------
    float
        Wind speed in m/s (1 kt = 0.514444 m/s), or NaN if input is NaN.
    """
    if pd.isna(speed_kt):
        return float("nan")
    return float(speed_kt) * 0.514444


def knots_to_kmh(speed_kt: float) -> float:
    """
    Convert wind speed from knots to km/h.

    Retained for backward compatibility.

    Parameters
    ----------
    speed_kt : float
        Wind speed in knots. NaN values are passed through.

    Returns
    -------
    float
        Wind speed in km/h (1 kt = 1.852 km/h), or NaN if input is NaN.
    """
    if pd.isna(speed_kt):
        return float("nan")
    return float(speed_kt) * 1.852


def knots_to_beaufort(speed_kt: float) -> int:
    """
    Convert wind speed in knots to the WMO Beaufort scale (0-12).

    Parameters
    ----------
    speed_kt : float
        Wind speed in knots. NaN values return N/A.

    Returns
    -------
    int
        Beaufort force number (0-12), or N/A if input is missing.

    Reference(s)
    ----------
    World Meteorological Organization (2023).
    Guide to Instruments and Methods of Observation,
    Volume I — Measurement of Meteorological Variables.
    WMO-No. 8, 2023 edition, Chapter 5 (Surface Wind).
    https://library.wmo.int/idurl/4/68695

    Examples
    --------
    >>> knots_to_beaufort(15.0)
    4
    >>> knots_to_beaufort(64.0)
    12
    """
    if pd.isna(speed_kt):
        return "N/A"
    kt = float(speed_kt)
    thresholds = [1, 4, 7, 11, 17, 22, 28, 34, 41, 48, 56, 64]
    for force, upper in enumerate(thresholds):
        if kt < upper:
            return force
    return 12


def knots_to_uv_barbs(speed_kt: float, direction_deg: float) -> Tuple[float, float]:
    """
    Compute MetPy-compatible U/V wind barb components.

    Convention: U = -v*sin(wdir), V = -v*cos(wdir).

    Parameters
    ----------
    speed_kt : float
        Wind speed in knots.
    direction_deg : float
        Wind direction in degrees (meteorological: direction wind is blowing
        FROM, measured clockwise from true north).

    Returns
    -------
    tuple of (float, float)
        (u_kt, v_kt) components suitable for ``ax.barbs()`` or
        ``StationPlot.plot_barb()``. Returns (NaN, NaN) if either input
        is missing.

    Examples
    --------
    >>> u, v = knots_to_uv_barbs(10.0, 270.0)   # westerly
    >>> round(u, 1), round(v, 1)
    (10.0, 0.0)
    """
    if pd.isna(speed_kt) or pd.isna(direction_deg):
        return float("nan"), float("nan")
    rad = math.radians(float(direction_deg))
    return -float(speed_kt) * math.sin(rad), -float(speed_kt) * math.cos(rad)


def inhg_to_hpa(pressure_inhg: float) -> float:
    """
    Convert atmospheric pressure from inches of mercury to hectopascals.

    Parameters
    ----------
    pressure_inhg : float
        Pressure in inches of mercury (inHg). NaN values are passed through.

    Returns
    -------
    float
        Pressure in hPa (1 inHg = 33.8639 hPa), or NaN if input is NaN.
    """
    if pd.isna(pressure_inhg):
        return float("nan")
    return float(pressure_inhg) * 33.8639


# [Sky display helper]

def sky_cover_from_code(code: str) -> int:
    """
    Convert a METAR sky cover code to oktas (0-8).

    Parameters
    ----------
    code : str
        METAR sky condition code (e.g. 'FEW', 'SCT', 'BKN', 'OVC', 'SKC').
        NaN / None values return 0 (clear).

    Returns
    -------
    int
        Sky cover in oktas (0-8).
    """
    if pd.isna(code):
        return 0
    return CLOUD_COVER_MAP.get(str(code).upper().strip()[:5], 0)


# [QC helpers]

def check_dewpoint_consistency(temp_c: float, dwpt_c: float) -> bool:
    """
    QC Check (Internal Consistency): Dewpoint <= Air Temperature.

    Physical basis: Td > T implies RH > 100 %, which is unphysical.

    Parameters
    ----------
    temp_c : float
        Air temperature in degrees Celsius.
    dwpt_c : float
        Dewpoint temperature in degrees Celsius.

    Returns
    -------
    bool
        True if the check passes (Td <= T) or either value is missing.

    Examples
    --------
    >>> check_dewpoint_consistency(16.0, 8.0)
    True
    >>> check_dewpoint_consistency(10.0, 12.0)
    False
    """
    if pd.isna(temp_c) or pd.isna(dwpt_c):
        return True
    return float(dwpt_c) <= float(temp_c)


def check_rh_range(relh: float) -> bool:
    """
    QC Check (Validity): Relative humidity within physical bounds [0, 100 %].

    Parameters
    ----------
    relh : float
        Relative humidity in percent. NaN values pass the check.

    Returns
    -------
    bool
        True if 0 <= relh <= 100 or the value is missing.

    Examples
    --------
    >>> check_rh_range(75.0)
    True
    >>> check_rh_range(105.0)
    False
    """
    if pd.isna(relh):
        return True
    return 0.0 <= float(relh) <= 100.0


def check_wind_temporal_spike(
    wspd_ms_now: float,
    wspd_ms_prev: float,
    hours_elapsed: float,
    max_change_ms_per_hour: float = 10.30,
) -> bool:
    """
    QC Check (Temporal): Wind speed change between consecutive reports.

    Flags an observation when the absolute wind speed change rate exceeds
    the threshold (default 10.30 m/s/hour, equivalent to 20 kt/hour).

    Both inputs must be in **m/s**. Returns True (pass) when either value
    is missing or the time interval is non-positive.

    Parameters
    ----------
    wspd_ms_now : float
        Wind speed of the current observation (m/s).
    wspd_ms_prev : float
        Wind speed of the preceding observation (m/s).
    hours_elapsed : float
        Time elapsed between the two observations (hours).
    max_change_ms_per_hour : float
        Rate-of-change threshold (m/s per hour). Default: 10.30.

    Examples
    --------
    >>> check_wind_temporal_spike(5.0, 3.0, 1.0)
    True
    >>> check_wind_temporal_spike(18.0, 3.0, 1.0)
    False
    """
    if pd.isna(wspd_ms_now) or pd.isna(wspd_ms_prev) or pd.isna(hours_elapsed):
        return True
    if float(hours_elapsed) <= 0:
        return True
    rate = abs(float(wspd_ms_now) - float(wspd_ms_prev)) / float(hours_elapsed)
    return rate <= max_change_ms_per_hour