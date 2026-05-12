#!/usr/bin/env python3
"""
Script Name: herbie_gfs.py
Purpose: Fetch GFS 0.25° analysis fields from the NOMADS/AWS/NCEI archive
         via herbie-data and return a single merged xarray.Dataset for a
         caller-specified target domain.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-05-12
Version: 1.0.0
License: MIT

Notes:
  • Context: Wraps herbie.Herbie to retrieve GFS analysis (fxx=0), merges
             the multi-level GRIB output into a flat Dataset,
             and subsets to the target domain. Realtime mode includes a
             cycle-latency guard so only published cycles are requested.
  • Inputs:  mode='realtime' | 'retrospective'; optional target_date (YYYY-MM-DD)
             and target_hour (0/6/12/18); required domain bounding box.
  • Outputs: xarray.Dataset with named variables (gh_850, t_850, u_850, v_850,
             gh_700, rh_700, u_700, v_700, gh_500, u_500, v_500, gh_250, u_250,
             v_250, prmsl, u_10m, v_10m) and attrs: valid_time, cycle, fxx, source.
  • Configuration: MODEL, PRODUCT, FXX, SEARCH_REGEX, GFS_CYCLE_HOURS,
                   CYCLE_LATENCY_HOURS, MAX_CYCLE_FALLBACKS are all tunable below.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import xarray as xr
from herbie import Herbie

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# [Save directory — GFS GRIB files cached here]
SAVE_DIR = Path(__file__).parent.parent / "data"

# [Model defaults]
MODEL   = "gfs"
PRODUCT = "pgrb2.0p25"
FXX     = 0

# [Search regex — all fields in one fetch]
SEARCH_REGEX = (
    r":(HGT|TMP|UGRD|VGRD):(850|700|500|250) mb"
    r"|:RH:700 mb"
    r"|:PRMSL:mean sea level"
    r"|:(UGRD|VGRD):10 m above ground"
    r"|:TMP:2 m above ground"
)

# [Cycle cadence]
GFS_CYCLE_HOURS     = (0, 6, 12, 18)
CYCLE_LATENCY_HOURS = 5    # cycles older than this (wall-clock UTC) are expected published

# [Realtime fallback]
MAX_CYCLE_FALLBACKS = 2    # try current, -6 h, -12 h before raising

# [Variable name map — (cfgrib shortName, typeOfLevel, level) -> dataset variable name]
_VAR_NAME_MAP = {
    ("gh",    "isobaricInhPa",   850): "gh_850",
    ("z",     "isobaricInhPa",   850): "gh_850",
    ("t",     "isobaricInhPa",   850): "t_850",
    ("u",     "isobaricInhPa",   850): "u_850",
    ("v",     "isobaricInhPa",   850): "v_850",
    ("gh",    "isobaricInhPa",   700): "gh_700",
    ("z",     "isobaricInhPa",   700): "gh_700",
    ("r",     "isobaricInhPa",   700): "rh_700",
    ("u",     "isobaricInhPa",   700): "u_700",
    ("v",     "isobaricInhPa",   700): "v_700",
    ("gh",    "isobaricInhPa",   500): "gh_500",
    ("z",     "isobaricInhPa",   500): "gh_500",
    ("u",     "isobaricInhPa",   500): "u_500",
    ("v",     "isobaricInhPa",   500): "v_500",
    ("gh",    "isobaricInhPa",   250): "gh_250",
    ("z",     "isobaricInhPa",   250): "gh_250",
    ("u",     "isobaricInhPa",   250): "u_250",
    ("v",     "isobaricInhPa",   250): "v_250",
    ("prmsl", "meanSea",           0): "prmsl",
    ("msl",   "meanSea",           0): "prmsl",
    ("mslet", "meanSea",           0): "prmsl",
    ("u",     "heightAboveGround", 10): "u_10m",
    ("v",     "heightAboveGround", 10): "v_10m",
    ("2t",    "heightAboveGround",  2): "t_2m",
    ("t",     "heightAboveGround",  2): "t_2m",
}

_LEVEL_COORD_CANDIDATES = ("level", "isobaricInhPa", "pressure_level")

_EXPECTED_VARS = {
    "gh_850", "t_850", "u_850", "v_850",
    "gh_700", "rh_700", "u_700", "v_700",
    "gh_500", "u_500", "v_500",
    "gh_250", "u_250", "v_250",
    "prmsl", "u_10m", "v_10m", "t_2m",
}

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("owl.herbie.gfs")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]

def _std_coords(da: xr.DataArray) -> xr.DataArray:
    """
    Rename non-standard lat/lon coord names and drop scalar non-spatial coords.

    Scalar auxiliary coords (e.g. heightAboveGround=10, heightAboveGround=2,
    meanSea=0) differ across variables and cause xr.Dataset(merged) to raise
    MergeError when they share the same coordinate name with different values.
    Dropping them here is safe because the variable name already encodes the
    level (u_10m, t_2m, prmsl, …).
    """
    rename = {}
    for c in da.coords:
        if str(c) in ("lat", "lats"):
            rename[str(c)] = "latitude"
        elif str(c) in ("lon", "lons"):
            rename[str(c)] = "longitude"
    if rename:
        da = da.rename(rename)
    spatial = {"latitude", "longitude"}
    to_drop = [c for c in da.coords if str(c) not in spatial and da.coords[c].ndim == 0]
    if to_drop:
        da = da.drop_vars(to_drop)
    return da

def _resolve_cycle(
    mode: str,
    target_date: str | None,
    target_hour: int | None,
) -> pd.Timestamp:
    """
    Resolve the GFS analysis cycle timestamp.

    In realtime mode, returns the most recent 00/06/12/18 UTC cycle whose
    age exceeds CYCLE_LATENCY_HOURS (to ensure data are published). In
    retrospective mode, validates and returns the requested date/hour.

    Parameters
    ----------
    mode : str
        ``'realtime'`` or ``'retrospective'``.
    target_date : str or None
        ISO date string (``'YYYY-MM-DD'``), required for retrospective mode.
    target_hour : int or None
        Analysis cycle hour (0, 6, 12, or 18), required for retrospective mode.

    Returns
    -------
    pd.Timestamp
        UTC timestamp of the resolved analysis cycle.
    """
    if mode == "retrospective":
        if target_date is None or target_hour is None:
            raise ValueError(
                "retrospective mode requires both target_date and target_hour"
            )
        if target_hour not in GFS_CYCLE_HOURS:
            raise ValueError(
                f"target_hour must be one of {GFS_CYCLE_HOURS}, got {target_hour}"
            )
        return pd.Timestamp(f"{target_date} {target_hour:02d}:00:00", tz="UTC")

    # realtime: find most recent cycle older than CYCLE_LATENCY_HOURS
    now    = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(hours=CYCLE_LATENCY_HOURS)

    valid_hours = [h for h in GFS_CYCLE_HOURS if h <= cutoff.hour]
    if valid_hours:
        candidate_hour = max(valid_hours)
        cycle = pd.Timestamp(
            year=cutoff.year, month=cutoff.month, day=cutoff.day,
            hour=candidate_hour, tz="UTC",
        )
    else:
        # All cycles of today are within latency window — use previous day's 18Z
        prev = cutoff - timedelta(days=1)
        cycle = pd.Timestamp(
            year=prev.year, month=prev.month, day=prev.day,
            hour=18, tz="UTC",
        )

    LOG.info("Realtime cycle resolved to %s", cycle)
    return cycle

def _fetch_with_fallback(
    cycle: pd.Timestamp,
    search: str,
    fxx: int = FXX,
) -> object:
    """
    Instantiate a Herbie object for *cycle* and retrieve fields matching *search*.

    On FileNotFoundError or OSError, retries up to MAX_CYCLE_FALLBACKS times
    with the previous 6-hour cycle, logging each attempt via LOG.warning.

    Parameters
    ----------
    cycle : pd.Timestamp
        UTC timestamp of the first cycle to attempt.
    search : str
        Herbie search string (regex over the GRIB index).
    fxx : int
        Forecast hour offset. Defaults to ``FXX`` (0 for analysis).

    Returns
    -------
    list or xr.Dataset
        Raw output from ``Herbie.xarray(search)``.
    """
    current = cycle
    for attempt in range(MAX_CYCLE_FALLBACKS + 1):
        try:
            h = Herbie(
                current.strftime("%Y-%m-%d %H:%M"),
                model=MODEL,
                product=PRODUCT,
                fxx=fxx,
                save_dir=SAVE_DIR,
            )
            LOG.info("Herbie cycle %s fxx=%d | source priority: %s", current, fxx, getattr(h, "priority", "default"))
            return h.xarray(search)
        except Exception as exc:
            if attempt < MAX_CYCLE_FALLBACKS:
                LOG.warning(
                    "Cycle %s unavailable (%s); falling back to previous cycle.",
                    current, exc,
                )
                current = current - pd.Timedelta(hours=6)
            else:
                raise RuntimeError(
                    f"GFS analysis unavailable after {MAX_CYCLE_FALLBACKS + 1} attempts. "
                    f"Last tried: {current}"
                ) from exc


def _merge_herbie_output(
    raw: object,
    cycle: pd.Timestamp,
) -> xr.Dataset:
    """
    Merge Herbie's multi-dataset output into a single flat xr.Dataset.

    Herbie returns one Dataset per GRIB typeOfLevel group. This function
    iterates each dataset, selects individual pressure levels (from a stacked
    ``level`` coordinate or from GRIB_level attributes), renames variables
    using ``_VAR_NAME_MAP``, and merges all into one Dataset.

    Parameters
    ----------
    raw : list or xr.Dataset
        Output from ``Herbie.xarray()``.
    cycle : pd.Timestamp
        Cycle timestamp; stored as a Dataset attribute.

    Returns
    -------
    xr.Dataset
        Flat dataset with standardized variable names.
    """
    if isinstance(raw, xr.Dataset):
        raw = [raw]

    merged: dict[str, xr.DataArray] = {}

    for ds in raw:
        level_coord = next(
            (c for c in _LEVEL_COORD_CANDIDATES if c in ds.coords), None
        )
        if level_coord is not None and ds[level_coord].ndim == 0:
            # ── One level per dataset (scalar coord) — use DataArray directly ──
            lev_int = int(ds[level_coord].values)
            for var in list(ds.data_vars):
                da    = ds[var].squeeze(drop=True)
                short = str(da.attrs.get("GRIB_shortName", var))
                key   = (short, "isobaricInhPa", lev_int)
                mapped = _VAR_NAME_MAP.get(key)
                if mapped is None:
                    continue
                merged[mapped] = _std_coords(da)
        elif level_coord is not None:
            # ── Multiple levels stacked along a 1-D coord — select each ─────
            for var in list(ds.data_vars):
                da_all = ds[var]
                short  = str(da_all.attrs.get("GRIB_shortName", var))
                for lev_val in ds[level_coord].values:
                    lev_int = int(lev_val)
                    key     = (short, "isobaricInhPa", lev_int)
                    mapped  = _VAR_NAME_MAP.get(key)
                    if mapped is None:
                        continue
                    da = da_all.sel({level_coord: lev_val}, drop=True).squeeze(drop=True)
                    merged[mapped] = _std_coords(da)
        else:
            # ── Single-level dataset (heightAboveGround, meanSea, …) ──────────
            for var in list(ds.data_vars):
                da       = ds[var].squeeze(drop=True)
                short    = str(da.attrs.get("GRIB_shortName", var))
                lev_type = str(da.attrs.get("GRIB_typeOfLevel", ""))
                try:
                    level = int(da.attrs.get("GRIB_level", -1))
                except (TypeError, ValueError):
                    level = -1

                key    = (short, lev_type, level)
                mapped = _VAR_NAME_MAP.get(key)

                if mapped is None:
                    # Alternative variable-name patterns for 10 m wind, MSLP, and 2 m T
                    if short in ("10u", "u10") or var == "u10":
                        mapped = "u_10m"
                    elif short in ("10v", "v10") or var == "v10":
                        mapped = "v_10m"
                    elif short in ("prmsl", "msl", "mslet"):
                        mapped = "prmsl"
                    elif (
                        short in ("2t", "t2m")
                        or var == "t2m"
                        or (short == "t" and lev_type == "heightAboveGround" and level == 2)
                    ):
                        mapped = "t_2m"
                    else:
                        LOG.warning("No mapping for var=%s short=%s type=%s lev=%d", var, short, lev_type, level)
                        continue

                merged[mapped] = _std_coords(da)

    if not merged:
        raise ValueError(
            "No variables could be mapped from the Herbie output. "
            "Check that herbie-data, cfgrib, and eccodes are installed."
        )

    missing = _EXPECTED_VARS - set(merged.keys())
    if missing:
        LOG.warning("Expected variables missing after merge: %s", missing)

    ds_out = xr.Dataset(merged)
    ds_out.attrs["cycle"] = str(cycle)
    ds_out.attrs["fxx"]   = FXX
    return ds_out


def _subset_to_domain(
    ds: xr.Dataset,
    domain: tuple[float, float, float, float],
) -> xr.Dataset:
    """
    Subset dataset to the requested bounding box.

    Converts GFS 0–360° longitudes to -180…180°, sorts ascending, then
    slices to the requested box. Handles descending-latitude axes.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``latitude`` and ``longitude`` coordinates (1-D).
    domain : tuple of (lon_min, lon_max, lat_min, lat_max)
        Bounding box in degrees (-180…180 longitudes).

    Returns
    -------
    xr.Dataset
        Domain-subset dataset.
    """
    lon_min, lon_max, lat_min, lat_max = domain

    # Convert 0–360 to -180…180 if necessary
    if float(ds["longitude"].max()) > 180:
        lon_new = xr.where(ds["longitude"] > 180, ds["longitude"] - 360, ds["longitude"])
        ds = ds.assign_coords(longitude=lon_new)
        ds = ds.sortby("longitude")

    # Sort latitude ascending if descending
    if float(ds["latitude"].values[0]) > float(ds["latitude"].values[-1]):
        ds = ds.sortby("latitude")

    ds = ds.sel(
        longitude=slice(lon_min, lon_max),
        latitude=slice(lat_min, lat_max),
    )
    return ds


def _assert_analysis(ds: xr.Dataset) -> None:
    """
    Assert that the dataset represents an analysis (fxx=0), not a forecast.

    Raises
    ------
    AssertionError
        If the fxx attribute is not zero.
    """
    fxx_attr = ds.attrs.get("fxx", -1)
    assert int(fxx_attr) == 0, (
        f"Expected analysis (fxx=0) but ds.attrs['fxx'] = {fxx_attr}. "
        "Ensure FXX=0 is set in herbie_gfs.py and the correct cycle was retrieved."
    )


# [fetch gfs analysis data]

def fetch_gfs_analysis(
    mode: str = "realtime",
    target_date: str | None = None,
    target_hour: int | None = None,
    domain: tuple[float, float, float, float] = None,
) -> xr.Dataset:
    """
    Fetch GFS 0.25° analysis fields for the target domain.

    A single Herbie call retrieves geopotential height, temperature, wind,
    relative humidity, MSLP, and 10 m wind for all requested levels. The 
    returned Dataset is subset to the target domain and verified to be an 
    analysis (fxx=0).

    Parameters
    ----------
    mode : str
        ``'realtime'``: most recent completed GFS cycle (with latency guard).
        ``'retrospective'``: cycle specified by *target_date* / *target_hour*.
    target_date : str or None
        ISO date string (``'YYYY-MM-DD'``), required when
        ``mode='retrospective'``.
    target_hour : int or None
        GFS cycle hour (0, 6, 12, or 18 UTC), required when
        ``mode='retrospective'``.
    domain : tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees. Required — set
        via the notebook configuration cell (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX).

    Returns
    -------
    xr.Dataset
        Dataset with variables: ``gh_850``, ``t_850``, ``u_850``, ``v_850``,
        ``gh_700``, ``rh_700``, ``u_700``, ``v_700``, ``gh_500``, ``u_500``,
        ``v_500``, ``gh_250``, ``u_250``, ``v_250``, ``prmsl``, ``u_10m``,
        ``v_10m``, ``t_2m`` (2 m temperature, K).
        Attrs: ``valid_time`` (UTC ISO string), ``cycle``, ``fxx``, ``source``.
    """
    if domain is None:
        raise ValueError(
            "domain is required: pass (lon_min, lon_max, lat_min, lat_max) "
            "from the notebook configuration cell."
        )
    cycle = _resolve_cycle(mode, target_date, target_hour)
    LOG.info("Fetching GFS analysis for cycle %s ...", cycle)

    raw = _fetch_with_fallback(cycle, SEARCH_REGEX, fxx=FXX)
    ds  = _merge_herbie_output(raw, cycle)
    ds  = _subset_to_domain(ds, domain)
    _assert_analysis(ds)

    ds.attrs["valid_time"] = str(cycle)
    ds.attrs["source"]     = "GFS 0.25° analysis via herbie-data"

    LOG.info(
        "Fetch complete: %d variables | valid_time=%s | domain=%s",
        len(ds.data_vars), ds.attrs["valid_time"], domain,
    )
    return ds
