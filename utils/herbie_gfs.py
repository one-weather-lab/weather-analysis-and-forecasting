#!/usr/bin/env python3
"""
Script Name: herbie_gfs.py
Purpose: Fetch GFS 0.25° analysis and forecast fields from the
         NOMADS/AWS/NCEI archive via herbie-data and return merged
         xarray.Datasets for a caller-specified target domain.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-05-24
Version: 1.9.3
License: MIT

"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cfgrib
import pandas as pd
import xarray as xr
from herbie import Herbie

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"cfgrib",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"This pattern is interpreted as a regular expression",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Will not remove GRIB file because it previously existed",
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# [Save directory — GFS GRIB files cached here]
SAVE_DIR = Path(__file__).parent.parent / "data"

# [Model defaults]
MODEL = "gfs"
FXX   = 0

# [Search regex — analysis (fetch_gfs_analysis) only; forecast uses _GFS_VAR_SEARCH]
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

# [Archive cutoff for product download fallback]
_DOWNLOAD_CUTOFF  = pd.Timestamp("2021-01-01", tz="UTC")

# [Herbie overwrite flag]
OVERWRITE = False

# [Parallel download workers]
MAX_WORKERS = 8

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

# [Per-variable search strings for forecast fetch — one GRIB message per (variable, fxx)]
_GFS_VAR_SEARCH: dict[str, str] = {
    "gh_850": ":HGT:850 mb:",
    "t_850":  ":TMP:850 mb:",
    "u_850":  ":UGRD:850 mb:",
    "v_850":  ":VGRD:850 mb:",
    "gh_700": ":HGT:700 mb:",
    "rh_700": ":RH:700 mb:",
    "u_700":  ":UGRD:700 mb:",
    "v_700":  ":VGRD:700 mb:",
    "gh_500": ":HGT:500 mb:",
    "u_500":  ":UGRD:500 mb:",
    "v_500":  ":VGRD:500 mb:",
    "gh_250": ":HGT:250 mb:",
    "u_250":  ":UGRD:250 mb:",
    "v_250":  ":VGRD:250 mb:",
}

_ANALYSIS_VARS = {
    "gh_850", "t_850", "u_850", "v_850",
    "gh_700", "rh_700", "u_700", "v_700",
    "gh_500", "u_500", "v_500",
    "gh_250", "u_250", "v_250",
    "prmsl", "u_10m", "v_10m", "t_2m",
}

_FORECAST_VARS = {
    "gh_850", "t_850", "u_850", "v_850",
    "gh_700", "rh_700", "u_700", "v_700",
    "gh_500", "u_500", "v_500",
    "gh_250", "u_250", "v_250",
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
# Suppress Herbie's per-file "✅ Found" INFO messages — too noisy in notebook context.
logging.getLogger("herbie").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]

def _gfs_product(cycle: pd.Timestamp) -> str:
    """
    Return the Herbie product string for *cycle*.

    Post-2021 (> ``_DOWNLOAD_CUTOFF``): ``'pgrb2.0p25'`` (NOMADS/AWS, 0.25-deg).
    Pre-2021 (≤ ``_DOWNLOAD_CUTOFF``): ``'0.5-degree'`` (NCEI half-degree archive).

    Parameters
    ----------
    cycle : pd.Timestamp
        UTC cycle timestamp.

    Returns
    -------
    str
        Herbie product identifier.
    """
    if cycle > _DOWNLOAD_CUTOFF:
        return "pgrb2.0p25"
    return "0.5-degree"


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
                product=_gfs_product(current),
                fxx=fxx,
                save_dir=SAVE_DIR,
                overwrite=OVERWRITE,
                verbose=False,
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


def _fetch_direct(
    cycle: pd.Timestamp,
    search: str,
    fxx: int,
    data_dir: Path,
) -> object:
    """
    Fetch GFS fields for specified cycle and fxx.

    For pre-2021 cycles the full GRIB is downloaded; ``OVERWRITE=False``
    skips re-download when the file is already present. For post-2021 cycles,
    ``OVERWRITE=False`` prevents subset re-downloads.

    Parameters
    ----------
    cycle : pd.Timestamp
        UTC cycle timestamp.
    search : str
        Herbie search string (regex over the GRIB index). Used only for
        post-2021 AWS fetches; ignored for pre-2021 full-GRIB reads.
    fxx : int
        Forecast lead time in hours.
    data_dir : Path
        Herbie download cache directory.

    Returns
    -------
    list of xr.Dataset or xr.Dataset
        Compatible with ``_merge_herbie_output``.
    """
    h = Herbie(
        cycle.strftime("%Y-%m-%d %H:%M"),
        model=MODEL,
        product=_gfs_product(cycle),
        fxx=fxx,
        save_dir=data_dir,
        overwrite=OVERWRITE,
        verbose=False,
    )
    LOG.info(
        "Herbie cycle %s fxx=%d | source: %s",
        cycle, fxx, getattr(h, "priority", "default"),
    )

    if cycle <= _DOWNLOAD_CUTOFF:
        h.download()
        return cfgrib.open_datasets(str(h.get_localFilePath()))

    return h.xarray(search)


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
                        if level == -1:
                            LOG.debug(
                                "Skipping unmapped var=%s short=%s type=%s (lev unparseable)",
                                var, short, lev_type,
                            )
                        else:
                            LOG.warning(
                                "No mapping for var=%s short=%s type=%s lev=%d",
                                var, short, lev_type, level,
                            )
                        continue

                merged[mapped] = _std_coords(da)

    if not merged:
        raise ValueError(
            "No variables could be mapped from the Herbie output. "
            "Check that herbie-data, cfgrib, and eccodes are installed."
        )

    missing = _ANALYSIS_VARS - set(merged.keys())
    if missing:
        LOG.warning("Analysis variables missing after merge: %s", missing)

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
    """
    fxx_attr = ds.attrs.get("fxx", -1)
    assert int(fxx_attr) == 0, (
        f"Expected analysis (fxx=0) but ds.attrs['fxx'] = {fxx_attr}. "
        "Ensure FXX=0 is set in herbie_gfs.py and the correct cycle was retrieved."
    )


def _build_lead_time_ds(
    raw: object,
    cycle: pd.Timestamp,
    domain: tuple[float, float, float, float],
) -> xr.Dataset:
    """
    Merge, subset, and filter raw Herbie output for one forecast lead time.

    Parameters
    ----------
    raw : list or xr.Dataset
        Output from ``_fetch_direct`` or ``cfgrib.open_datasets``.
    cycle : pd.Timestamp
        UTC cycle timestamp passed to ``_merge_herbie_output``.
    domain : tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)`` bounding box.

    Returns
    -------
    xr.Dataset
        Domain-subset Dataset containing only ``_FORECAST_VARS``.
    """
    ds = _merge_herbie_output(raw, cycle)
    ds = _subset_to_domain(ds, domain)
    drop_vars = [v for v in ds.data_vars if v not in _FORECAST_VARS]
    if drop_vars:
        ds = ds.drop_vars(drop_vars)
    return ds


def _download_lead_time(
    cycle: pd.Timestamp,
    variable: str | None,
    fxx: int,
    data_dir: Path,
) -> None:
    """
    Download the GFS GRIB for one forecast lead time without reading it.

    Pre-2021 cycles: downloads the full GRIB (``h.download()``) to the flat
    ``YYYYMMDD/`` directory. Post-2021 cycles: downloads a single-variable
    subset GRIB (``h.download(_GFS_VAR_SEARCH[variable])``) to
    ``YYYYMMDD/{variable}/``. ``OVERWRITE=False`` skips re-download when the
    file is already present.

    Parameters
    ----------
    cycle : pd.Timestamp
        UTC cycle timestamp.
    variable : str or None
        Variable key in ``_GFS_VAR_SEARCH`` (post-2021), or ``None``
        for pre-2021 full-GRIB downloads.
    fxx : int
        Forecast lead time in hours.
    data_dir : Path
        Herbie download cache directory.
    """
    if cycle <= _DOWNLOAD_CUTOFF:
        h = Herbie(
            cycle.strftime("%Y-%m-%d %H:%M"),
            model=MODEL,
            product=_gfs_product(cycle),
            fxx=fxx,
            save_dir=data_dir,
            overwrite=OVERWRITE,
            verbose=False,
        )
        LOG.info(
            "Herbie cycle %s fxx=%d | source: %s",
            cycle, fxx, getattr(h, "priority", "default"),
        )
        h.download()
    else:
        var_dir = data_dir / "gfs" / cycle.strftime("%Y%m%d") / variable
        var_dir.mkdir(parents=True, exist_ok=True)
        h = Herbie(
            cycle.strftime("%Y-%m-%d %H:%M"),
            model=MODEL,
            product=_gfs_product(cycle),
            fxx=fxx,
            save_dir=data_dir,
            overwrite=OVERWRITE,
            verbose=False,
        )
        LOG.info(
            "Herbie cycle %s fxx=%d var=%s | source: %s",
            cycle, fxx, variable, getattr(h, "priority", "default"),
        )
        local_path = h.download(_GFS_VAR_SEARCH[variable])
        if local_path is None:
            raise ValueError(
                f"Search '{_GFS_VAR_SEARCH[variable]}' matched no messages "
                f"for var={variable} fxx={fxx}"
            )
        local_path = Path(local_path)
        dest = var_dir / local_path.name
        if local_path.exists() and not dest.exists():
            local_path.rename(dest)



def _grib_glob(
    date_dir: Path,
    cycle: pd.Timestamp,
    hz: str,
    fxx: int,
    variable: str | None = None,
) -> list:
    """
    Return glob matches for a GFS GRIB file at the specified lead time.

    Pre-2021 NCEI files are in the flat ``YYYYMMDD/`` directory following
    ``gfs_4_{YYYYMMDD}_{HH}00_{FFF}.grb2``. Post-2021 NOMADS/AWS files are
    in per-variable subdirectories ``YYYYMMDD/{variable}/`` following
    ``[subset_HASH__]gfs.t{HH}z.pgrb2.0p25.f{FFF}``.

    Parameters
    ----------
    date_dir : Path
        The date directory (``{data_dir}/gfs/YYYYMMDD/``) to search.
    cycle : pd.Timestamp
        UTC cycle timestamp; selects naming convention.
    hz : str
        Two-digit zero-padded cycle hour string (``'00'``, ``'06'``, etc.).
    fxx : int
        Forecast lead time in hours.
    variable : str or None
        Variable name for post-2021 per-variable subdirectory lookup.
        Ignored for pre-2021 data.

    Returns
    -------
    list of Path
        All matching paths (empty list if none found).
    """
    if cycle <= _DOWNLOAD_CUTOFF:
        date_str = cycle.strftime("%Y%m%d")
        return list(date_dir.glob(f"gfs_4_{date_str}_{hz}00_{fxx:03d}.grb2"))
    search_dir = (date_dir / variable) if variable else date_dir
    if not search_dir.exists():
        return []
    return list(search_dir.glob(f"*gfs.t{hz}z*.f{fxx:03d}"))


# [Resolve GFS data folder]

def resolve_gfs_data(
    mode: str,
    target_date: str | None = None,
    target_hour: int | None = None,
    horizon_h: int = 48,
    step_h: int = 6,
    data_dir: Path = None,
) -> tuple[pd.Timestamp, bool]:
    """
    Resolve the GFS cycle and check whether all forecast GRIBs are cached.

    Post-2021 cycles: checks for one GRIB per ``(variable, fxx)`` pair in
    ``{data_dir}/gfs/YYYYMMDD/{variable}/`` (one subdirectory per variable
    in ``_GFS_VAR_SEARCH``). Pre-2021 NCEI cycles: checks for one full GRIB
    per lead time in the flat ``{data_dir}/gfs/YYYYMMDD/`` directory.

    Parameters
    ----------
    mode : str
        ``'realtime'`` or ``'retrospective'``.
    target_date : str or None
        ISO date string (``'YYYY-MM-DD'``), required when
        ``mode='retrospective'``.
    target_hour : int or None
        GFS cycle hour (0, 6, 12, or 18 UTC), required when
        ``mode='retrospective'``.
    horizon_h : int
        Maximum forecast lead time in hours (e.g., 48).
    step_h : int
        Lead-time increment in hours (e.g., 6).
    data_dir : Path or None
        Root data directory. Defaults to ``data/`` relative to the module.

    Returns
    -------
    tuple of (pd.Timestamp, bool)
        Resolved UTC cycle timestamp and a flag that is ``True`` only when
        every ``(variable, fxx)`` pair (post-2021) or every lead time
        (pre-2021) has a matching GRIB file on disk.
    """
    if data_dir is None:
        data_dir = SAVE_DIR
    data_dir   = Path(data_dir)
    cycle      = _resolve_cycle(mode, target_date, target_hour)
    date_dir   = data_dir / "gfs" / cycle.strftime("%Y%m%d")
    lead_times = list(range(0, horizon_h + 1, step_h))
    hz         = f"{cycle.hour:02d}"

    if not date_dir.exists():
        LOG.info(
            "0/%d files found for cycle %s (t%sz) — date directory absent (%s)",
            len(lead_times), cycle, hz, date_dir,
        )
        return cycle, False

    if cycle > _DOWNLOAD_CUTOFF:
        expected = len(_GFS_VAR_SEARCH) * len(lead_times)
        found = sum(
            1 for var in _GFS_VAR_SEARCH
            for fxx in lead_times
            if _grib_glob(date_dir, cycle, hz, fxx, variable=var)
        )
        LOG.info(
            "%d/%d (variable×fxx) files found for cycle %s (t%sz) in %s",
            found, expected, cycle, hz, date_dir,
        )
    else:
        expected = len(lead_times)
        found = sum(
            1 for fxx in lead_times
            if _grib_glob(date_dir, cycle, hz, fxx)
        )
        LOG.info(
            "%d/%d files found for cycle %s (t%sz) in %s",
            found, expected, cycle, hz, date_dir,
        )
    return cycle, found == expected


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
        Attrs: ``valid_time`` (UTC ISO string), ``cycle``, ``fxx``.
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

    LOG.info(
        "Fetch complete: %d variables | valid_time=%s | domain=%s",
        len(ds.data_vars), ds.attrs["valid_time"], domain,
    )
    return ds


# [fetch gfs forecast data]

def fetch_gfs_forecast(
    mode: str = "realtime",
    target_date: str | None = None,
    target_hour: int | None = None,
    horizon_h: int = 120,
    step_h: int = 6,
    data_dir: Path = None,
    max_workers: int = MAX_WORKERS,
) -> pd.Timestamp:
    """
    Download GFS GRIBs for all lead times in the forecast horizon.

    Post-2021 cycles: downloads one GRIB per ``(variable, fxx)`` pair to
    ``{data_dir}/gfs/YYYYMMDD/{variable}/``. Pre-2021 cycles: downloads one
    full GRIB per lead time to the flat ``YYYYMMDD/`` directory.

    Parameters
    ----------
    mode : str
        ``'realtime'``: most recent completed GFS cycle (with latency guard).
        ``'retrospective'``: cycle specified by ``target_date`` /
        ``target_hour``.
    target_date : str or None
        ISO date string (``'YYYY-MM-DD'``), required when
        ``mode='retrospective'``.
    target_hour : int or None
        GFS cycle hour (0, 6, 12, or 18 UTC), required when
        ``mode='retrospective'``.
    horizon_h : int
        Maximum forecast lead time in hours (e.g., 48).
    step_h : int
        Lead-time increment in hours (e.g., 6 for 6-hourly output).
        fxx=0 (analysis at cycle time) is always included as the first step.
    data_dir : Path
        Herbie download cache directory. Defaults to ``data/`` relative to the
        module location if not provided.
    max_workers : int
        Number of concurrent download threads. Defaults to ``MAX_WORKERS``
        (8). Reduce if NOMADS rate-limits or connection errors are observed.

    Returns
    -------
    pd.Timestamp
        Resolved UTC cycle timestamp; pass directly to ``build_gfs_forecast_ds``.
    """
    if data_dir is None:
        data_dir = SAVE_DIR
    data_dir = Path(data_dir)

    cycle = _resolve_cycle(mode, target_date, target_hour)
    lead_times = list(range(0, horizon_h + 1, step_h))
    LOG.info(
        "Downloading GFS forecast | cycle=%s | step=%d h | horizon=%d h | workers=%d",
        cycle, step_h, horizon_h, max_workers,
    )

    if cycle > _DOWNLOAD_CUTOFF:
        jobs = [(var, fxx) for var in _GFS_VAR_SEARCH for fxx in lead_times]

        def _dl(job: tuple[str, int]) -> None:
            var, fxx = job
            _download_lead_time(cycle, var, fxx, data_dir)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_dl, job): job for job in jobs}
            for future in as_completed(futures):
                var, fxx = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    LOG.warning("var=%s fxx=%d: download failed — %s", var, fxx, exc)

        date_dir = data_dir / "gfs" / cycle.strftime("%Y%m%d")
        hz = f"{cycle.hour:02d}"
        missing = [
            (var, fxx) for var in _GFS_VAR_SEARCH for fxx in lead_times
            if not _grib_glob(date_dir, cycle, hz, fxx, variable=var)
        ]
        if missing:
            LOG.warning(
                "GFS forecast: %d file(s) absent after parallel fetch — retrying sequentially",
                len(missing),
            )
            for var, fxx in missing:
                try:
                    _download_lead_time(cycle, var, fxx, data_dir)
                except Exception as exc:
                    LOG.warning("Retry failed for var=%s fxx=%d: %s", var, fxx, exc)
    else:
        def _dl(fxx: int) -> None:
            _download_lead_time(cycle, None, fxx, data_dir)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_dl, fxx): fxx for fxx in lead_times}
            for future in as_completed(futures):
                fxx = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    LOG.warning("fxx=%d: download failed — %s", fxx, exc)

    LOG.info("GFS forecast download complete | cycle=%s", cycle)
    return cycle


def build_gfs_forecast_ds(
    cycle: pd.Timestamp,
    horizon_h: int = 48,
    step_h: int = 6,
    domain: tuple[float, float, float, float] = None,
    data_dir: Path = None,
) -> xr.Dataset:
    """
    Assemble the GFS forecast Dataset from cached GRIBs.

    Post-2021 cycles: reads one per-variable GRIB per lead time from
    ``{data_dir}/gfs/YYYYMMDD/{variable}/``, decodes each single-message
    file directly, and merges all variables into one Dataset per lead time.
    Pre-2021 NCEI cycles: reads one full GRIB per lead time from the flat
    ``{data_dir}/gfs/YYYYMMDD/`` directory via ``_build_lead_time_ds``.
    Both paths concatenate slices along the ``valid_time`` dimension.

    Parameters
    ----------
    cycle : pd.Timestamp
        Resolved UTC cycle timestamp (returned by ``fetch_gfs_forecast``).
    horizon_h : int
        Maximum forecast lead time in hours (e.g., 48).
    step_h : int
        Lead-time increment in hours (e.g., 6 for 6-hourly output).
    domain : tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
    data_dir : Path
        Herbie download cache directory. Defaults to ``data/`` relative to the
        module location if not provided.

    Returns
    -------
    xr.Dataset
        Variables: ``gh_850``, ``t_850``, ``u_850``, ``v_850``,
        ``gh_700``, ``rh_700``, ``u_700``, ``v_700``,
        ``gh_500``, ``u_500``, ``v_500``,
        ``gh_250``, ``u_250``, ``v_250``.
        Coordinate ``valid_time`` holds lead times in hours from init.
        Attrs: ``init_date``.
    """
    if domain is None:
        raise ValueError("domain is required: pass (lon_min, lon_max, lat_min, lat_max)")
    if data_dir is None:
        data_dir = SAVE_DIR
    data_dir = Path(data_dir)

    date_dir   = data_dir / "gfs" / cycle.strftime("%Y%m%d")
    hz         = f"{cycle.hour:02d}"
    lead_times = list(range(0, horizon_h + 1, step_h))
    LOG.info(
        "Building GFS forecast Dataset | cycle=%s | %d lead times | domain=%s",
        cycle.strftime('%Y-%m-%d %H:%M UTC'), len(lead_times), domain,
    )

    slices: list[xr.Dataset] = []
    valid_leads: list[int] = []

    for fxx in lead_times:
        if cycle > _DOWNLOAD_CUTOFF:
            merged: dict[str, xr.DataArray] = {}
            for var in sorted(_FORECAST_VARS):
                matches = _grib_glob(date_dir, cycle, hz, fxx, variable=var)
                if not matches:
                    raise FileNotFoundError(
                        f"GRIB not found for var={var} fxx={fxx} in "
                        f"{date_dir / var}. Call fetch_gfs_forecast() first."
                    )
                dsets = cfgrib.open_datasets(str(matches[0]))
                if not dsets:
                    raise ValueError(
                        f"cfgrib returned empty list for var={var} fxx={fxx}: "
                        f"{matches[0]}"
                    )
                ds_var = dsets[0]
                raw_name = list(ds_var.data_vars)[0]
                da = ds_var[raw_name].squeeze(drop=True)
                merged[var] = _std_coords(da)
            ds = xr.Dataset(merged)
            ds = _subset_to_domain(ds, domain)
        else:
            matches = _grib_glob(date_dir, cycle, hz, fxx)
            grib_path = matches[0] if matches else None
            if grib_path is None:
                raise FileNotFoundError(
                    f"GRIB not found for fxx={fxx} in {date_dir}. "
                    "Call fetch_gfs_forecast() first."
                )
            raw = cfgrib.open_datasets(str(grib_path))
            ds = _build_lead_time_ds(raw, cycle, domain)
        slices.append(ds)
        valid_leads.append(fxx)

    combined = xr.concat(
        slices,
        dim=pd.Index(valid_leads, name="valid_time"),
    )
    combined.attrs["init_date"] = cycle.strftime('%Y-%m-%d %H:%M UTC')

    LOG.info(
        "GFS forecast Dataset complete: %d lead times | vars=%s",
        len(valid_leads), list(combined.data_vars),
    )
    return combined
