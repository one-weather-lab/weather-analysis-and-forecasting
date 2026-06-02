#!/usr/bin/env python3
"""
Script Name: herbie_gefs.py
Purpose: Fetch GEFS ensemble member forecast fields via herbie-data; parallel
         over members; cached to data/.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-06-02
Version: 1.6.9
License: MIT
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie

from herbie_gfs import _resolve_cycle, _std_coords, _subset_to_domain

_ECCODES_LOCK = threading.Lock()

warnings.filterwarnings("ignore", category=FutureWarning, module=r"cfgrib")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# [Save directory — GEFS GRIB files cached here]
SAVE_DIR = Path(__file__).parent.parent / "data"

# [GEFS model defaults]
GEFS_MODEL        = "gefs"
GEFS_PRODUCT_25   = "atmos.25"   # [0.25-deg; GEFSv12, available from 2020-09-23 onwards]
GEFS_PRODUCT_5    = "atmos.5"    # [0.50-deg; only option for pre-GEFSv12 archives]
_GEFS_V12_CUTOFF  = pd.Timestamp("2020-09-23")   # GEFSv12 operational date

# [Variable search strings]
_GEFS_SEARCH = {
    "mslp"          : r":PRMSL:mean sea level:",
    "t2m"           : r":TMP:2 m above ground:",
    "precip_6h"     : r":APCP:surface",
    "gust10m"       : r":GUST:surface:",
    "cape"          : r":CAPE:surface:",
    "cin"           : r":CIN:surface:",
    "tmax_2m"       : r":TMAX:2 m above ground:",
    "tmin_2m"       : r":TMIN:2 m above ground:",
    # [SNOD eccodes shortName unverified — check GRIB_shortName after first download]
    "snod"          : r":SNOD:surface:",
}

# [Period-based variables absent from pgrb2s at fxx=0 — no accumulation/max at init]
_SFLUX_ABSENT_AT_FXX0 = {"precip_6h", "tmax_2m", "tmin_2m"}

# [Temperature-extreme variables stored in Kelvin in the GRIB — converted to degC on build]
_KELVIN_VARS = {"tmax_2m", "tmin_2m"}

# [Expected cfgrib shortNames per scalar variable — verify each entry on first live run]
_GEFS_VALID_NAMES: dict[str, set[str]] = {
    "mslp"          : {"prmsl"},
    "t2m"           : {"2t"},
    "precip_6h"     : {"tp"},
    "gust10m"       : {"gust"},
    "cape"          : {"cape"},
    "cin"           : {"cin"},
    "tmax_2m"       : {"tmax"},
    "tmin_2m"       : {"tmin"},
}   

# [Parallel download workers]
MAX_WORKERS = 8

# [Quality control thresholds]
_MIN_MEMBERS_REQUIRED = 10

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("owl.herbie.gefs")
# Suppress Herbie's per-file "✅ Found" INFO messages — too noisy in notebook context.
logging.getLogger("herbie").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]

def _gefs_product(cycle: pd.Timestamp) -> str:
    """
    Return the Herbie GEFS product string for *cycle*.

    Parameters
    ----------
    cycle : pd.Timestamp
        UTC cycle timestamp (tz-aware or tz-naive).

    Returns
    -------
    str
        ``GEFS_PRODUCT_25`` for GEFSv12 cycles (>= 2020-09-23);
        ``GEFS_PRODUCT_5`` for earlier archives.
    """
    c = cycle.replace(tzinfo=None) if cycle.tzinfo is not None else cycle
    return GEFS_PRODUCT_25 if c >= _GEFS_V12_CUTOFF else GEFS_PRODUCT_5


def _member_str(member_id: int) -> str:
    """Format a member index as the Herbie member string (0 → 'c00'; 1–30 → 'p01'–'p30')."""
    return "c00" if member_id == 0 else f"p{member_id:02d}"


def _extract_da(
    raw: object,
    variable: str,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Extract a scalar DataArray or a (u, v) pair from Herbie's raw xarray output.

    Parameters
    ----------
    raw : list or xr.Dataset
        Output from ``Herbie.xarray()``.
    variable : str
        Variable key from ``_GEFS_SEARCH``; governs U/V splitting for wind keys.

    Returns
    -------
    xr.DataArray
        For scalar variables: single DataArray with standardized coords.
    tuple of (xr.DataArray, xr.DataArray)
        For wind keys: ``(u_component, v_component)``.
    """
    datasets = [raw] if isinstance(raw, xr.Dataset) else list(raw)

    das: list[xr.DataArray] = []
    for ds in datasets:
        for vname in ds.data_vars:
            da = ds[vname].squeeze(drop=True)
            das.append(_std_coords(da))

    if not das:
        raise ValueError(f"No data arrays extracted for variable '{variable}'")

    if variable in _WIND_KEYS:
        u_das, v_das = [], []
        for da in das:
            short = str(da.attrs.get("GRIB_shortName", ""))
            if short in ("u", "10u", "ugrd"):
                u_das.append(da)
            elif short in ("v", "10v", "vgrd"):
                v_das.append(da)
        if not u_das or not v_das:
            raise ValueError(
                f"Could not separate U/V for variable '{variable}'; "
                f"found short names: {[str(da.attrs.get('GRIB_shortName','?')) for da in das]}"
            )
        return u_das[0], v_das[0]

    valid_names = _GEFS_VALID_NAMES.get(variable)
    if valid_names is None:
        return das[0]

    for da in das:
        short = str(da.attrs.get("GRIB_shortName", "")).lower()
        if short in valid_names:
            return da

    found_shorts = [str(d.attrs.get("GRIB_shortName", "?")) for d in das]
    raise ValueError(
        f"No array in file matches {valid_names} for variable '{variable}'; "
        f"found shortNames: {found_shorts}"
    )


def _download_member_at_lead(
    cycle: pd.Timestamp,
    member_id: int,
    fxx: int,
    variable: str,
    data_dir: Path,
) -> None:
    """
    Download a single GEFS member GRIB at one lead time without decoding.

    Parameters
    ----------
    cycle : pd.Timestamp
        UTC cycle timestamp.
    member_id : int
        Member index (0 = control c00; 1–30 = perturbed p01–p30).
    fxx : int
        Lead time in hours.
    variable : str
        Variable key from ``_GEFS_SEARCH``.
    data_dir : Path
        Root data cache directory. Files are written to
        ``{data_dir}/gefs/{YYYYMMDD}/{variable}/``.
    """
    if fxx == 0 and variable in _SFLUX_ABSENT_AT_FXX0:
        return

    search = _GEFS_SEARCH[variable]
    try:
        h = Herbie(
            cycle.strftime("%Y-%m-%d %H:%M"),
            model=GEFS_MODEL,
            product=_gefs_product(cycle),
            member=_member_str(member_id),
            fxx=fxx,
            save_dir=data_dir,
            verbose=False,
        )
        local_path = h.download(search)
        if local_path is None:
            raise ValueError(
                f"Search '{search}' matched no messages for "
                f"{_member_str(member_id)} fxx={fxx}"
            )
        local_path = Path(local_path)
        var_dir = data_dir / "gefs" / cycle.strftime("%Y%m%d") / variable
        var_dir.mkdir(parents=True, exist_ok=True)
        dest = var_dir / local_path.name
        if local_path.exists() and not dest.exists():
            local_path.rename(dest)
    except Exception as exc:
        LOG.warning(
            "GEFS %s member=%s fxx=%d download failed: %s",
            variable, _member_str(member_id), fxx, exc,
        )


def _stack_lead_time(
    fxx_results: dict[int, object],
    valid_members: list[int],
    variable: str,
    domain: tuple[float, float, float, float],
    is_wind: bool,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Concatenate per-member results at one lead time and subset to domain.

    Parameters
    ----------
    fxx_results : dict
        Mapping member_id → DataArray or (u_da, v_da) or None.
    valid_members : list of int
        Member IDs for which the fetch succeeded.
    variable : str
        Variable key; determines wind vs scalar path.
    domain : tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)`` bounding box.
    is_wind : bool
        True when variable requires U/V splitting.

    Returns
    -------
    xr.DataArray or tuple of (xr.DataArray, xr.DataArray)
        Stacked and domain-subset result.
    """
    member_idx = pd.Index(valid_members, name="member")

    if is_wind:
        u_das = [fxx_results[mid][0] for mid in valid_members]
        v_das = [fxx_results[mid][1] for mid in valid_members]
        u_stack = xr.concat(u_das, dim=member_idx)
        v_stack = xr.concat(v_das, dim=member_idx)
        u_stack = _subset_to_domain(xr.Dataset({"u": u_stack}), domain)["u"]
        v_stack = _subset_to_domain(xr.Dataset({"v": v_stack}), domain)["v"]
        return u_stack, v_stack

    das = [fxx_results[mid] for mid in valid_members]
    stack = xr.concat(das, dim=member_idx)
    stack = _subset_to_domain(xr.Dataset({"var": stack}), domain)["var"]
    return stack


def _grib_glob_gefs(
    date_dir: Path,
    cycle: pd.Timestamp,
    member_id: int,
    fxx: int,
) -> list:
    """
    Return glob matches for a single GEFS member GRIB at the specified lead time.

   GEFS files are locally stored in ``{data_dir}/gefs/{YYYYMMDD}/`` (flat structure,
    consistent with the GFS cache layout). File naming follows NOAA conventions:
    ``[subset_{hash}__]ge{member}.t{HH}z.pgrb2s.0p25.f{FFF}``
    where member is ``c00`` (control) or ``p01``–``p30`` (perturbed).

    Parameters
    ----------
    date_dir : Path
        The date directory (``{data_dir}/gefs/{YYYYMMDD}/``) to search.
    cycle : pd.Timestamp
        UTC cycle timestamp; used to extract the two-digit cycle hour.
    member_id : int
        Member index (0 = control c00; 1–30 = perturbed p01–p30).
    fxx : int
        Forecast lead time in hours.

    Returns
    -------
    list of Path
        All matching paths (empty list if none found).
    """
    hz = f"{cycle.hour:02d}"
    member = _member_str(member_id)
    return list(date_dir.glob(f"*ge{member}.t{hz}z*.f{fxx:03d}"))


# [Resolve GEFS data folder]

def resolve_gefs_data(
    mode: str,
    target_date: str | None = None,
    target_hour: int | None = None,
    horizon_h: int = 48,
    data_dir: Path = None,
    variable: str | None = None,
) -> tuple[pd.Timestamp, bool]:
    """
    Resolve the GEFS cycle and check whether the requested variable's GRIBs are already
    locally stored.

    Herbie stores GEFS files in ``{data_dir}/gefs/{YYYYMMDD}/{variable}/``
    when ``variable`` is provided, or ``{data_dir}/gefs/{YYYYMMDD}/`` when
    ``variable`` is None. Files are counted across all 31 possible member IDs (0–30) for each
    expected lead time. Variables in ``_SFLUX_ABSENT_AT_FXX0`` have no GRIB
    at fxx=0; the count skips that slot to avoid a false cache miss.

    Parameters
    ----------
    mode : str
        ``'realtime'`` or ``'retrospective'``.
    target_date : str or None
        ISO date string (``'YYYY-MM-DD'``), required when
        ``mode='retrospective'``.
    target_hour : int or None
        GEFS cycle hour (0, 6, 12, or 18 UTC), required when
        ``mode='retrospective'``.
    horizon_h : int
        Maximum forecast lead time in hours (e.g., 48).
    data_dir : Path or None
        Root data directory. Defaults to ``SAVE_DIR``.
    variable : str or None
        Variable key from ``_GEFS_SEARCH``. When provided, the cache flag is
        ``True`` only when the file count threshold is met AND the spot-check
        confirms the variable is decodable.

    Returns
    -------
    tuple of (pd.Timestamp, bool)
        Resolved UTC cycle timestamp and a flag that is ``True`` only when
        the file count threshold is met (and the variable spot-check passes
        when ``variable`` is provided).
    """
    if data_dir is None:
        data_dir = SAVE_DIR
    data_dir = Path(data_dir)

    cycle      = _resolve_cycle(mode, target_date, target_hour)
    date_dir   = (
        data_dir / "gefs" / cycle.strftime("%Y%m%d") / variable
        if variable is not None
        else data_dir / "gefs" / cycle.strftime("%Y%m%d")
    )
    lead_times = list(range(0, horizon_h + 1, 6))

    if not date_dir.exists():
        LOG.info(
            "0 files found for cycle %s — directory absent (%s)",
            cycle, date_dir,
        )
        return cycle, False

    _lead_to_count = (
        lead_times[1:] if variable in _SFLUX_ABSENT_AT_FXX0 else lead_times
    )

    # Count over all 31 possible member IDs (0 = control; 1–30 = perturbed).
    n_found = sum(
        1
        for mid in range(31)
        for fxx in _lead_to_count
        if _grib_glob_gefs(date_dir, cycle, mid, fxx)
    )
    LOG.info(
        "%d files found for cycle %s in %s (minimum: %d)",
        n_found, cycle, date_dir, _MIN_MEMBERS_REQUIRED * len(_lead_to_count),
    )

    if n_found < _MIN_MEMBERS_REQUIRED * len(_lead_to_count):
        return cycle, False

    if variable is not None:
        spot_fxx = lead_times[1] if variable in _SFLUX_ABSENT_AT_FXX0 else lead_times[0]
        sample   = _grib_glob_gefs(date_dir, cycle, 0, spot_fxx)
        _valid   = _GEFS_VALID_NAMES.get(variable)
        _filters = [{"shortName": sn} for sn in _valid] if _valid else [{}]
        _found   = False
        for path in sample:
            for _fby in _filters:
                try:
                    with _ECCODES_LOCK:
                        raw = cfgrib.open_datasets(str(path), filter_by_keys=_fby)
                    if raw:
                        _found = True
                        break
                except Exception as _exc:
                    LOG.debug(
                        "GEFS spot-check %s %s fby=%s: %s",
                        variable, path.name, _fby, _exc,
                    )
            if _found:
                break
        if not _found:
            LOG.info(
                "'%s' not decodable at c00 fxx=%d — fetch required",
                variable, spot_fxx,
            )
            return cycle, False

    return cycle, True


def fetch_gefs_members(
    mode: str = "realtime",
    target_date: str | None = None,
    target_hour: int | None = None,
    horizon_h: int = 120,
    variable: str = None,
    data_dir: Path = None,
    max_workers: int = MAX_WORKERS,
    *,
    n_members: int,
) -> pd.Timestamp:
    """
    Download GEFS GRIBs for all members across the forecast horizon.

    Parameters
    ----------
    mode : str
        ``'realtime'``: most recent completed GEFS cycle (with latency guard).
        ``'retrospective'``: cycle specified by ``target_date`` / ``target_hour``.
    target_date : str or None
        ISO date string (``'YYYY-MM-DD'``), required when
        ``mode='retrospective'``.
    target_hour : int or None
        GEFS cycle hour (0, 6, 12, or 18 UTC), required when
        ``mode='retrospective'``.
    horizon_h : int
        Maximum forecast lead time in hours; downloads fxx = 0, 6, 12, …, horizon_h.
    variable : str
        Variable identifier; must be a key in ``_GEFS_SEARCH``.
    data_dir : Path
        Herbie download cache directory. Defaults to ``SAVE_DIR``.
    max_workers : int
        Thread count for parallel downloads. Defaults to ``MAX_WORKERS``.
        Reduce if NOMADS rate-limits or connection errors are observed.
    n_members : int
        Number of GEFS members to download starting from member 0 (control).
        Member IDs 0 (c00) through ``n_members - 1`` (p{n_members-1}) are
        fetched. Maximum 31 (1 control + 30 perturbed).

    Returns
    -------
    pd.Timestamp
        Resolved UTC cycle timestamp; pass directly to ``build_gefs_forecast_da``.
    """
    if variable is None or variable not in _GEFS_SEARCH:
        raise ValueError(
            f"variable must be one of {list(_GEFS_SEARCH)}, got {variable!r}"
        )
    if data_dir is None:
        data_dir = SAVE_DIR
    data_dir = Path(data_dir)

    cycle      = _resolve_cycle(mode, target_date, target_hour)
    lead_times = list(range(0, horizon_h + 1, 6))
    member_ids = list(range(n_members))
    tasks      = [(mid, fxx) for fxx in lead_times for mid in member_ids]

    LOG.info(
        "Downloading GEFS %s | cycle=%s | horizon=%d h | members=%d | workers=%d",
        variable, cycle, horizon_h, n_members, max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_member_at_lead, cycle, mid, fxx, variable, data_dir): (mid, fxx)
            for mid, fxx in tasks
        }
        for future in as_completed(futures):
            mid, fxx = futures[future]
            try:
                future.result()
            except Exception as exc:
                LOG.warning("fxx=%d member=%s: download failed — %s", fxx, _member_str(mid), exc)

    LOG.info("GEFS %s download complete | cycle=%s", variable, cycle)
    return cycle


def build_gefs_forecast_da(
    cycle: pd.Timestamp,
    variable: str,
    domain: tuple[float, float, float, float],
    horizon_h: int = 48,
    data_dir: Path = None,
    return_uv: bool = False,
    *,
    n_members: int,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Assemble the GEFS DataArray from already-downloaded GRIBs.

    Parameters
    ----------
    cycle : pd.Timestamp
        Resolved UTC cycle timestamp (returned by ``fetch_gefs_members``).
    variable : str
        Variable identifier; must be a key in ``_GEFS_SEARCH``.
    domain : tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees (−180…180).
    horizon_h : int
        Maximum forecast lead time in hours.
    data_dir : Path
        Herbie download cache directory. Defaults to ``SAVE_DIR``.
    return_uv : bool
        If True, variable must be a wind key; returns ``(u_da, v_da)``
        instead of a single DataArray.
    n_members : int
        Number of GEFS members to read starting from member 0 (control).
        Must match the value used in the corresponding ``fetch_gefs_members``
        call.

    Returns
    -------
    xr.DataArray
        Dims ``(member, valid_time, latitude, longitude)``. Attrs include
        ``variable``, ``init_date``.
    tuple of (xr.DataArray, xr.DataArray)
        Only when ``return_uv=True``: ``(u_component, v_component)``,
        same dims.
    """
    if variable not in _GEFS_SEARCH:
        raise ValueError(
            f"variable must be one of {list(_GEFS_SEARCH)}, got {variable!r}"
        )
    if return_uv and variable not in _WIND_KEYS:
        raise ValueError(
            f"return_uv=True requires a wind variable ({_WIND_KEYS}), got {variable!r}"
        )
    if domain is None:
        raise ValueError("domain is required: pass (lon_min, lon_max, lat_min, lat_max)")
    if data_dir is None:
        data_dir = SAVE_DIR
    data_dir = Path(data_dir)

    date_dir   = data_dir / "gefs" / cycle.strftime("%Y%m%d") / variable
    lead_times = list(range(0, horizon_h + 1, 6))
    member_ids = list(range(n_members))
    is_wind    = variable in _WIND_KEYS

    LOG.info(
        "Building GEFS %s DataArray | cycle=%s | %d lead times | domain=%s",
        variable, cycle.strftime('%Y-%m-%d %H:%M UTC'), len(lead_times), domain,
    )

    time_slices:   list[xr.DataArray] = []
    time_slices_u: list[xr.DataArray] = []
    time_slices_v: list[xr.DataArray] = []

    for fxx in lead_times:
        if fxx == 0 and variable in _SFLUX_ABSENT_AT_FXX0 and not is_wind:
            time_slices.append(None)
            continue

        fxx_results: dict[int, object] = {}
        for mid in member_ids:
            matches = _grib_glob_gefs(date_dir, cycle, mid, fxx)
            if not matches:
                LOG.warning(
                    "GEFS %s member=%s fxx=%d: no GRIB in %s — member skipped",
                    variable, _member_str(mid), fxx, date_dir,
                )
                fxx_results[mid] = None
                continue

            extracted = None
            for path in matches:
                try:
                    with _ECCODES_LOCK:
                        _snames = _GEFS_VALID_NAMES.get(variable)
                        if _snames and not is_wind:
                            _raw: list = []
                            for sn in _snames:
                                _raw.extend(
                                    cfgrib.open_datasets(
                                        str(path), filter_by_keys={"shortName": sn}
                                    )
                                )
                            raw = _raw
                        else:
                            raw = cfgrib.open_datasets(str(path))
                    extracted = _extract_da(raw, variable)
                    break
                except Exception as _exc:
                    LOG.debug(
                        "GEFS %s candidate %s: %s",
                        variable, path.name, _exc,
                    )
                    continue
            if extracted is None:
                LOG.warning(
                    "GEFS %s member=%s fxx=%d: variable not found in any cached GRIB",
                    variable, _member_str(mid), fxx,
                )
            fxx_results[mid] = extracted

        valid = [mid for mid in member_ids if fxx_results.get(mid) is not None]
        if len(valid) < _MIN_MEMBERS_REQUIRED:
            raise RuntimeError(
                f"Only {len(valid)} members decoded for variable={variable!r} fxx={fxx}; "
                f"minimum required is {_MIN_MEMBERS_REQUIRED}."
            )
        if len(valid) < n_members:
            LOG.warning(
                "fxx=%d: %d/%d members for %s; proceeding with available members.",
                fxx, len(valid), n_members, variable,
            )

        stacked = _stack_lead_time(fxx_results, valid, variable, domain, is_wind)

        if is_wind:
            time_slices_u.append(stacked[0])
            time_slices_v.append(stacked[1])
        else:
            time_slices.append(stacked)

    if time_slices and time_slices[0] is None:
        if len(time_slices) < 2 or time_slices[1] is None:
            raise RuntimeError(
                f"Cannot synthesize fxx=0 fill for {variable!r}: fxx=6 slice unavailable"
            )
        if variable in _KELVIN_VARS:
            time_slices[0] = xr.full_like(time_slices[1], float("nan"))
        else:
            time_slices[0] = xr.zeros_like(time_slices[1])
        LOG.info(
            "GEFS %s fxx=0: zero-filled (period-based variable absent at initialization)",
            variable,
        )

    lead_idx = pd.Index(lead_times, name="valid_time")
    init_str  = cycle.strftime('%Y-%m-%d %H:%M UTC')

    if is_wind:
        u_da = xr.concat(time_slices_u, dim=lead_idx).transpose(
            "member", "valid_time", "latitude", "longitude"
        )
        v_da = xr.concat(time_slices_v, dim=lead_idx).transpose(
            "member", "valid_time", "latitude", "longitude"
        )
        u_da.attrs.update({"variable": variable, "component": "u", "init_date": init_str})
        v_da.attrs.update({"variable": variable, "component": "v", "init_date": init_str})
        LOG.info("GEFS %s (U+V) DataArray complete: shape=%s", variable, dict(u_da.sizes))
        return (u_da, v_da) if return_uv else u_da

    combined = xr.concat(time_slices, dim=lead_idx).transpose(
        "member", "valid_time", "latitude", "longitude"
    )
    if variable in _KELVIN_VARS:
        combined = combined - 273.15
    combined.attrs.update({"variable": variable, "init_date": init_str})
    LOG.info("GEFS %s DataArray complete: shape=%s", variable, dict(combined.sizes))
    return combined
