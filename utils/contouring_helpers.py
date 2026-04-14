#!/usr/bin/env python3
"""
Script Name: contouring_helpers.py
Purpose: Objective analysis utilities for gridding scattered surface
         observations onto a regular mesh and smoothing the resulting
         fields for synoptic-scale contouring.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-04-13
Version: 1.0.0
License: MIT

Notes:
  • Context: Intermediate processing step between data fetching/decoding
             (iem_raw, noaa_realtime, metar_helpers) and map rendering
             (plot_helpers). Produces gridded fields consumed by all
             isobar, isotherm, and upper-air contour functions.
  • Inputs:  Decoded, QC-passed METAR DataFrame with columns ``station``,
             ``lat``, ``lon``, ``mslp`` (hPa), ``temp_c`` (°C),
             ``dwpt_c`` (°C).  Coordinate arrays from ``build_europe_grid()``.
  • Outputs: 2-D numpy arrays suitable for ``plt.contour`` / ``plt.contourf``.
  • Configuration: Default grid resolution and smoothing sigmas are defined
                   as function parameters; the notebook's Configuration cell
                   provides the authoritative values at run time.

  Standard grid schema
  --------------------
  ``grid_lon``  : 1-D numpy array, ascending, degrees East
  ``grid_lat``  : 1-D numpy array, ascending, degrees North
  ``data``      : 2-D numpy array, shape ``(len(grid_lat), len(grid_lon))``
                  — row index corresponds to latitude,
                    column index corresponds to longitude.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("owl.contouring.helpers")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Grid construction]

def build_europe_grid(
    lon_min: float = -25,
    lon_max: float = 45,
    lat_min: float = 30,
    lat_max: float = 72,
    resolution: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create regularly spaced 1-D longitude and latitude arrays.

    The arrays are suitable for ``np.meshgrid`` and for passing to
    ``grid_variable()`` as the target grid.

    Parameters
    ----------
    lon_min : float
        Western boundary (degrees East).
    lon_max : float
        Eastern boundary (degrees East).
    lat_min : float
        Southern boundary (degrees North).
    lat_max : float
        Northern boundary (degrees North).
    resolution : float
        Grid spacing in degrees.  0.25° ≈ 25 km at mid-latitudes,
        appropriate for the typical METAR network density over Europe.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(grid_lon, grid_lat)`` — both 1-D, ascending.
    """
    grid_lon = np.arange(lon_min, lon_max + resolution / 2, resolution)
    grid_lat = np.arange(lat_min, lat_max + resolution / 2, resolution)
    return grid_lon, grid_lat

# [Interpolation and smoothing]

def grid_variable(
    obs_lons: np.ndarray,
    obs_lats: np.ndarray,
    obs_values: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Interpolate irregularly spaced observations onto a regular grid.

    Wraps ``scipy.interpolate.griddata``.  The output array has shape
    ``(len(grid_lat), len(grid_lon))`` — row = latitude, column = longitude.

    Parameters
    ----------
    obs_lons : np.ndarray
        1-D array of observation longitudes (degrees East).
    obs_lats : np.ndarray
        1-D array of observation latitudes (degrees North).
    obs_values : np.ndarray
        1-D array of observed values (same length as *obs_lons*).
    grid_lon : np.ndarray
        1-D target longitude array (ascending).
    grid_lat : np.ndarray
        1-D target latitude array (ascending).
    method : str
        Interpolation method passed to ``griddata``.  One of
        ``'nearest'``, ``'linear'``, ``'cubic'``.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(grid_lat), len(grid_lon))``.
    """
    obs_lons = np.asarray(obs_lons).ravel()
    obs_lats = np.asarray(obs_lats).ravel()
    obs_values = np.asarray(obs_values).ravel()

    if obs_lons.ndim != 1 or obs_lats.ndim != 1 or obs_values.ndim != 1:
        raise ValueError("obs_lons, obs_lats, and obs_values must be 1-D.")
    if not (len(obs_lons) == len(obs_lats) == len(obs_values)):
        raise ValueError(
            f"Length mismatch: obs_lons={len(obs_lons)}, "
            f"obs_lats={len(obs_lats)}, obs_values={len(obs_values)}."
        )

    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)
    return griddata(
        (obs_lons, obs_lats),
        obs_values,
        (mesh_lon, mesh_lat),
        method=method,
    )

def smooth_grid(
    data_2d: np.ndarray,
    sigma: float = 6,
) -> np.ndarray:
    """
    Apply NaN-safe Gaussian smoothing to a 2-D field.

    NaN values are temporarily filled with the field mean before
    filtering, and the original NaN mask is restored afterwards.
    This prevents NaN propagation through the convolution kernel.

    Parameters
    ----------
    data_2d : np.ndarray
        2-D array to smooth (shape ``(n_lat, n_lon)``).
    sigma : float
        Standard deviation of the Gaussian kernel in grid points.

    Returns
    -------
    np.ndarray
        Smoothed 2-D array (same shape as input).
    """
    nan_mask = np.isnan(data_2d)
    filled = data_2d.copy()
    filled[nan_mask] = np.nanmean(data_2d)

    smoothed = gaussian_filter(filled, sigma=sigma)
    smoothed[nan_mask] = np.nan
    return smoothed

# [Composite field gridder]

def grid_surface_fields(
    df: pd.DataFrame,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    sigma_mslp: float = 6,
    sigma_t: float = 4,
    sigma_td: float = 4,
) -> dict[str, np.ndarray]:
    """
    Grid and smooth MSLP, temperature, and dewpoint in one call.

    Each variable is handled independently: NaN rows are dropped
    per variable so that a station missing pressure does not
    contaminate the temperature grid (no cross-contamination).

    Parameters
    ----------
    df : pd.DataFrame
        Decoded observation table.  Must contain columns
        ``lon``, ``lat``, ``mslp`` (hPa),
        ``temp_c`` (°C), ``dwpt_c`` (°C).
    grid_lon : np.ndarray
        1-D target longitude array from ``build_europe_grid()``.
    grid_lat : np.ndarray
        1-D target latitude array from ``build_europe_grid()``.
    sigma_mslp : float
        Gaussian smoothing sigma for sea-level pressure.
    sigma_t : float
        Gaussian smoothing sigma for temperature.
    sigma_td : float
        Gaussian smoothing sigma for dewpoint.

    Returns
    -------
    dict of str -> np.ndarray
        Keys: ``'mslp'``, ``'temperature'``, ``'dewpoint'``.
        Each value is a 2-D array of shape
        ``(len(grid_lat), len(grid_lon))``.
    """
    result = {}

    # ── MSLP ──────────────────────────────────────────────────────────────
    mask_mslp = df[["lon", "lat", "mslp"]].dropna()
    LOG.info("[GRID] MSLP: %d valid observations", len(mask_mslp))
    raw_mslp = grid_variable(
        mask_mslp["lon"].values,
        mask_mslp["lat"].values,
        mask_mslp["mslp"].values,
        grid_lon, grid_lat,
    )
    result["mslp"] = smooth_grid(raw_mslp, sigma=sigma_mslp)

    # ── Temperature ───────────────────────────────────────────────────────
    mask_t = df[["lon", "lat", "temp_c"]].dropna()
    LOG.info("[GRID] Temperature: %d valid observations", len(mask_t))
    raw_t = grid_variable(
        mask_t["lon"].values,
        mask_t["lat"].values,
        mask_t["temp_c"].values,
        grid_lon, grid_lat,
    )
    result["temperature"] = smooth_grid(raw_t, sigma=sigma_t)

    # ── Dewpoint ──────────────────────────────────────────────────────────
    mask_td = df[["lon", "lat", "dwpt_c"]].dropna()
    LOG.info("[GRID] Dewpoint: %d valid observations", len(mask_td))
    raw_td = grid_variable(
        mask_td["lon"].values,
        mask_td["lat"].values,
        mask_td["dwpt_c"].values,
        grid_lon, grid_lat,
    )
    result["dewpoint"] = smooth_grid(raw_td, sigma=sigma_td)

    return result


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Example:")
    print("  from contouring_helpers import build_europe_grid, grid_surface_fields")
