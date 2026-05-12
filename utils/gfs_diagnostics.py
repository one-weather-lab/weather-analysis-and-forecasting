#!/usr/bin/env python3
"""
Script Name: gfs_diagnostics.py
Purpose: Thin MetPy wrappers for computing synoptic-scale upper-air
         diagnostics from GFS DataArrays.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-05-12
Version: 1.0.0
License: MIT

Notes:
  • Context: All functions accept xarray.DataArray inputs extracted from the
             Dataset produced by herbie_gfs.fetch_gfs_analysis(). Input DataArrays
             carry cfgrib-derived unit attributes.
  • Inputs:  xarray.DataArray objects with 1-D 'latitude' and 'longitude'
             coordinates (ascending, degrees), as returned by herbie_gfs.py.
  • Outputs: xarray.DataArray objects with pint units detached via
             .metpy.dequantify() and long_name / units attrs for direct plotting.
  • Configuration: No tunable module-level constants; all logic is parameterised
                   through function arguments.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging

import numpy as np
import xarray as xr
import metpy.calc as mpcalc

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("owl.gfs.diagnostics")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# [Internal helpers]

def _build_dx_dy(
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> tuple:
    """
    Build 2-D dx/dy spacing arrays from 1-D latitude and longitude.

    Parameters
    ----------
    lat : xr.DataArray
        1-D latitude array (degrees North, ascending).
    lon : xr.DataArray
        1-D longitude array (degrees East, ascending).

    Returns
    -------
    tuple of (lat_2d, lon_2d, dx, dy)
        ``lat_2d``, ``lon_2d``: 2-D numpy arrays (shape ny × nx).
        ``dx``, ``dy``: pint Quantity arrays of horizontal spacing (metres).
    """
    lon_vals = np.asarray(lon.values, dtype=float)
    lat_vals = np.asarray(lat.values, dtype=float)
    lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)
    dx, dy = mpcalc.lat_lon_grid_deltas(lon_2d, lat_2d)
    return lat_2d, lon_2d, dx, dy

# [Diagnostic functions]

def compute_temperature_advection(
    u: xr.DataArray,
    v: xr.DataArray,
    T: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> xr.DataArray:
    """
    Compute temperature advection in °C/s.

    Parameters
    ----------
    u, v : xr.DataArray
        Zonal and meridional wind components at the analysis level (m/s).
    T : xr.DataArray
        Air temperature at the analysis level (K).
    lat, lon : xr.DataArray
        1-D coordinate arrays (degrees).

    Returns
    -------
    xr.DataArray
        Temperature advection (°C/s) with coordinates preserved.
    """
    lat_2d, lon_2d, dx, dy = _build_dx_dy(lat, lon)

    u_q = u.metpy.quantify()
    v_q = v.metpy.quantify()
    T_q = T.metpy.quantify().metpy.convert_units("degC")

    T_adv = mpcalc.advection(T_q, u=u_q, v=v_q, dx=dx, dy=dy)
    T_adv = T_adv.metpy.dequantify()
    T_adv = T_adv.assign_attrs(long_name="Temperature advection", units="°C s**-1")
    LOG.info("Temperature advection computed: shape=%s", T_adv.shape)
    return T_adv


def compute_relative_vorticity(
    u: xr.DataArray,
    v: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> xr.DataArray:
    """
    Compute relative vorticity from the analyzed wind field.

    Parameters
    ----------
    u, v : xr.DataArray
        Zonal and meridional wind components at the analysis level (m/s).
    lat, lon : xr.DataArray
        1-D coordinate arrays (degrees).

    Returns
    -------
    xr.DataArray
        Relative vorticity (1/s) with coordinates preserved.
    """
    _, _, dx, dy = _build_dx_dy(lat, lon)

    u_q = u.metpy.quantify()
    v_q = v.metpy.quantify()

    rvort = mpcalc.vorticity(u_q, v_q, dx=dx, dy=dy)
    rvort = rvort.metpy.dequantify()
    rvort = rvort.assign_attrs(long_name="Relative vorticity", units="s**-1")
    LOG.info("Relative vorticity computed: shape=%s", rvort.shape)
    return rvort


def compute_relative_vorticity_advection(
    u: xr.DataArray,
    v: xr.DataArray,
    rvort: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> xr.DataArray:
    """
    Compute relative vorticity advection.

    Parameters
    ----------
    u, v : xr.DataArray
        Zonal and meridional wind components at the analysis level (m/s).
    rvort : xr.DataArray
        Relative vorticity (1/s) from ``compute_relative_vorticity()``.
    lat, lon : xr.DataArray
        1-D coordinate arrays (degrees).

    Returns
    -------
    xr.DataArray
        Relative vorticity advection (1/s²) with coordinates preserved.
    """
    _, _, dx, dy = _build_dx_dy(lat, lon)

    u_q = u.metpy.quantify()
    v_q = v.metpy.quantify()
    z_q = rvort.metpy.quantify()

    rvort_adv = mpcalc.advection(z_q, u=u_q, v=v_q, dx=dx, dy=dy)
    rvort_adv = rvort_adv.metpy.dequantify()
    rvort_adv = rvort_adv.assign_attrs(
        long_name="Relative vorticity advection", units="s**-2"
    )
    LOG.info("Relative vorticity advection computed: shape=%s", rvort_adv.shape)
    return rvort_adv


def compute_wind_speed(
    u: xr.DataArray,
    v: xr.DataArray,
) -> xr.DataArray:
    """
    Compute total horizontal wind speed from u and v components.

    Parameters
    ----------
    u, v : xr.DataArray
        Zonal and meridional wind components (m/s).

    Returns
    -------
    xr.DataArray
        Horizontal wind speed (m/s) with original coordinates preserved.
    """
    u_q  = u.metpy.quantify()
    v_q  = v.metpy.quantify()
    wspd = mpcalc.wind_speed(u_q, v_q)
    wspd = wspd.metpy.dequantify()
    wspd = wspd.assign_attrs(long_name="Wind speed", units="m s**-1")
    return wspd
