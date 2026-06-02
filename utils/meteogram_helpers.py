#!/usr/bin/env python3
"""
Script Name: meteogram_helpers.py
Purpose: Nearest-grid-point identification and three-panel ensemble meteogram for
         the point-based ensemble forecast section.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-06-02
Version: 1.3.1
License: MIT

Notes:
  • Context: Supports Section 3 of 05_synoptic_forecasting. Provides (1) haversine
             nearest-grid-point lookup and (2) a three-panel plume/box-plot meteogram.
  • Inputs:  POI coordinates, GEFS DataArrays at the nearest grid point, lead times.
  • Outputs: Path objects for saved PNGs.
  • Configuration: _EARTH_RADIUS_KM is a module-level constant;
                   dpi and output_dir are caller-specified.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
_EARTH_RADIUS_KM = 6371.009   # WGS84 mean radius (IUGG 2015)

# Meteogram percentile envelopes
_P_IQR_LO, _P_IQR_HI      = 25, 75   # inner shaded band
_P_ENV_LO, _P_ENV_HI      = 10, 90   # outer shaded band
_CONTROL_MEMBER            = 0        # index of the control member in the member dim

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("owl.meteogram")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]

def _haversine_km(
    lat1: float, lon1: float,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """
    Compute great-circle distances from (lat1, lon1) to all (lat2, lon2) points.

    Parameters
    ----------
    lat1, lon1 : float
        Origin coordinates (degrees).
    lat2, lon2 : np.ndarray
        Target latitudes and longitudes (degrees); broadcast-compatible shapes.

    Returns
    -------
    np.ndarray
        Great-circle distances in kilometres.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _build_plume_panel(
    ax: plt.Axes,
    data: np.ndarray,
    valid_times: np.ndarray,
    label: str,
    unit: str,
    color: str,
) -> None:
    """
    Draw a plume panel: thin member trajectories, IQR shading, 10–90 % envelope,
    control member overlay, and summary statistics annotation.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    data : np.ndarray
        Array of shape (n_members, n_times).
    valid_times : np.ndarray
        Lead times in hours (1-D).
    label : str
        Panel title label.
    unit : str
        Y-axis unit string.
    color : str
        Base colour for shading.
    """
    n_members = data.shape[0]

    p10 = np.percentile(data, _P_ENV_LO, axis=0)
    p25 = np.percentile(data, _P_IQR_LO, axis=0)
    p75 = np.percentile(data, _P_IQR_HI, axis=0)
    p90 = np.percentile(data, _P_ENV_HI, axis=0)

    # Thin member trajectories
    for i in range(n_members):
        ax.plot(valid_times, data[i], color=color, alpha=0.12, linewidth=0.6)

    # 10–90 % envelope
    ax.fill_between(valid_times, p10, p90, alpha=0.18, color=color, label="10–90 %")
    # IQR
    ax.fill_between(valid_times, p25, p75, alpha=0.40, color=color, label="IQR (25–75 %)")

    # Control member (index 0)
    ax.plot(
        valid_times, data[_CONTROL_MEMBER],
        color="black", linewidth=1.4, linestyle="--", label="Control (p00)",
    )

    mean_val   = float(np.mean(data))
    std_val    = float(np.std(data))

    annotation = f"Mean: {mean_val:.1f} {unit}  |  Std: {std_val:.1f}"
    ax.annotate(
        annotation,
        xy=(0.01, 0.02), xycoords="axes fraction",
        fontsize=10, color="dimgrey",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
    )

    ax.set_ylabel(f"{label} {unit}", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3, linestyle=":")


def _build_precip_panel(
    ax: plt.Axes,
    data: np.ndarray,
    valid_times: np.ndarray,
) -> None:
    """
    Draw a precipitation box-plot panel with one box per 6-hour step.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    data : np.ndarray
        Array of shape (n_members, n_times).
    valid_times : np.ndarray
        Lead times in hours (1-D).
    """
    box_data = [data[:, i] for i in range(data.shape[1])]

    bp = ax.boxplot(
        box_data,
        positions=valid_times,
        widths=3.0,
        whis=[_P_ENV_LO, _P_ENV_HI],
        patch_artist=True,
        medianprops=dict(color="navy", linewidth=1.5),
        boxprops=dict(facecolor="#cce5ff", alpha=0.8),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
        manage_ticks=False,
    )
    ax.set_ylabel("6 h precipitation [ mm ]", fontsize=12)
    ax.set_xlim(valid_times[0] - 5, valid_times[-1] + 5)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3, linestyle=":", axis="y")


# [Find nearest grid point]

def get_nearest_grid_point(
    poi_lat: float,
    poi_lon: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> tuple[tuple[int, int], float]:
    """
    Find the nearest grid point to a point of interest using haversine distance.

    Parameters
    ----------
    poi_lat, poi_lon : float
        Point-of-interest coordinates (degrees).
    lat_grid, lon_grid : np.ndarray
        1-D arrays of grid latitudes and longitudes (degrees).

    Returns
    -------
    idx : tuple of (int, int)
        ``(row, column)`` index of the nearest grid point.
    distance_km : float
        Great-circle distance to the nearest grid point (km).
    """
    lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
    dist_2d = _haversine_km(poi_lat, poi_lon, lat_2d, lon_2d)

    flat_idx = int(np.argmin(dist_2d))
    row = flat_idx // lon_2d.shape[1]
    col = flat_idx  % lon_2d.shape[1]

    nearest_lat = float(lat_grid[row])
    nearest_lon = float(lon_grid[col])
    distance_km = float(dist_2d[row, col])

    LOG.info(
        "Nearest grid point: (%s, %s) → (%.2f N, %.2f E) | dist=%.1f km",
        row, col, nearest_lat, nearest_lon, distance_km,
    )
    return (row, col), distance_km

# [Build meteogram]

def build_meteogram(
    t2m: xr.DataArray,
    precip: xr.DataArray,
    gust: xr.DataArray,
    valid_times: np.ndarray,
    poi_name: str,
    init_date: str,
    distance_km: float,
    output_dir: str | Path,
    dpi: int = 150,
) -> Path:
    """
    Construct a three-panel ensemble meteogram.

    Panel 1: T2m plume with all members as thin trajectories; IQR (25th–75th
    percentile) shaded; 10th–90th percentile envelope; control member as a
    distinct line; summary statistics annotation.

    Panel 2: Precipitation box plots with one box per 6-hour step; box = 25th–75th
    percentile, whiskers = 10th–90th percentile, outliers as individual points.

    Panel 3: Wind gust plume with same percentile structure as Panel 1.

    Unit conversions are applied internally:
      - T2m:  K   -> degC  (subtract 273.15)
      - Gust: m/s -> km/h  (multiply 3.6)
      - Precipitation: kg/m2 == mm (no conversion)

    Parameters
    ----------
    t2m : xr.DataArray
        2-meter air temperature, dims ``(member, valid_time)``, units K (GEFS raw).
    precip : xr.DataArray
        6-hour accumulated precipitation, dims ``(member, valid_time)``, units mm.
    gust : xr.DataArray
        10-meter wind gust, dims ``(member, valid_time)``, units m/s (GEFS raw).
    valid_times : np.ndarray
        Lead times in hours from ``init_date`` (1-D integer array).
    poi_name : str
        Location name for the figure title.
    init_date : str
        Initialization timestamp string (``str(pd.Timestamp)``); used for
        title formatting and x-tick computation.
    distance_km : float
        Great-circle distance from the POI to the nearest GEFS grid point (km);
        displayed in the figure title.
    output_dir : str or Path
        Directory for the saved PNG. Created if absent.
    dpi : int
        Output resolution (dots per inch).

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unit conversions (GEFS raw -> display units)
    t2m_np    = np.asarray(t2m.values)   - 273.15   # K -> degC
    precip_np = np.asarray(precip.values)             # kg/m2 == mm
    gust_np   = np.asarray(gust.values)   * 3.6      # m/s -> km/h
    times_np  = np.asarray(valid_times, dtype=float)

    # Formatted init time and actual valid-time tick labels
    init_ts      = pd.Timestamp(init_date)
    init_str_fmt = init_ts.strftime("%Y-%m-%d %H:%M UTC")
    tick_labels  = [
        (init_ts + pd.Timedelta(hours=int(h))).strftime("%d/%m\n%H:%M")
        for h in times_np
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False, sharey=False)
    fig.suptitle(
        f"Ensemble Meteogram — {poi_name}\n"
        f"Init: {init_str_fmt}  |  Grid point {distance_km:.1f} km from POI",
        fontsize=13, y=0.93,
    )

    # Panel 1: T2m plume
    _build_plume_panel(axes[0], t2m_np, times_np, "2 m temperature", "[ °C ]", "#e74c3c")

    # Panel 2: Precipitation box plots
    _build_precip_panel(axes[1], precip_np, times_np)

    # Panel 3: Wind gust plume
    _build_plume_panel(axes[2], gust_np, times_np, "10 m wind gust", "[ km/h ]", "#2980b9")

    # Replace numeric lead-time ticks with actual valid-time labels on all panels;
    # x-axis label on bottom panel only.
    for ax in axes:
        ax.set_xticks(times_np)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis="x", labelsize=11)
    axes[2].set_xlabel("Valid time (UTC)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"meteogram_{timestamp}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    plt.close(fig)

    return out_path.resolve()
