#!/usr/bin/env python3
"""
Script Name: plot_helpers.py
Purpose: Surface and upper-air map visualizations for the European domain.
         Renders station models, isobar/isotherm contours, and GFS upper-air
         fields (850, 700, 500, and 250 hPa; surface analysis) as publication-quality
         PNGs. Covers both Greece-scale and continental-scale domains.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-05-25
Version: 3.3.2
License: MIT
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import io
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import logging

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from metpy.calc import reduce_point_density
from metpy.plots import SkewT
from metpy.units import units

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# [Data sources]
AIRPORTS_URL = (
    "https://raw.githubusercontent.com/davidmegginson/"
    "ourairports-data/master/airports.csv"
)

OURAIRPORTS_FULL_URL = (
    "https://davidmegginson.github.io/ourairports-data/airports.csv"
)

VALID_AIRPORT_TYPES = {"large_airport", "medium_airport"}   # expand to include "small_airport" if needed

# [Country]
ISO_COUNTRY = "GR"        # Two-letter ISO country code for network filtering

# [Map geometry]
GREECE_EXTENT  = [19.0, 29.5, 34.5, 42.0]   # [lon_min, lon_max, lat_min, lat_max]
EUROPE_EXTENT = [-25, 45, 30, 72]   # [lon_min, lon_max, lat_min, lat_max]

# [Figure output]
FIG_SIZE_IN    = (20, 20)    # figure size in inches
FIG_DPI        = 300         # output DPI for saved PNG

# [Wind rendering]
CALM_THRESHOLD_KT = 1        # speed (kt) below which wind is considered calm
BARB_LENGTH        = 10      # barb length in points
VRB_SCATTER_MIN_S  = 8       # minimum scatter marker size for VRB wind
VRB_SCATTER_MAX_S  = 60      # maximum scatter marker size for VRB wind
CALM_CIRCLE_S      = 36     # scatter marker size for calm wind circles

# [Station display]
THINNING_RADIUS_KM      = 200  # point-density thinning for standard station plot
T_THINNING_KM           = 0    # tighter thinning for temperature+wind overlay
RAOB_THINNING_RADIUS_KM = 150  # point-density thinning for 500 hPa RAOB station plot

# [Label layout]
LABEL_LON_OFFSET = 0.08     # longitude offset for T / RH text labels (degrees)
LABEL_LAT_UPPER  = 0.10     # latitude offset upward for temperature label
LABEL_LAT_LOWER  = 0.10     # latitude offset downward for RH label

# [Font sizes]
FONT_LABEL   = 18            # temperature and RH text
FONT_COUNT   = 18            # station count annotation
FONT_TITLE   = 22            # map title

# [Font scale factors — relative to FONT_LABEL]
FONT_SCALE_STATION  = 0.80   # T / RH labels on continental network plot (≈ 6 pt when FONT_LABEL=15)
FONT_SCALE_MMSLP_RAW  = 0.53   # raw MSLP station text  (≈ 8 pt when FONT_LABEL=15)
FONT_SCALE_CONTOUR  = 0.60   # isobar / isotherm inline labels (≈ 9 pt)
FONT_SCALE_UPPER_TITLE  = 0.75    # title font scale relative to FONT_TITLE
FONT_SCALE_UPPER_CBAR   = 0.80    # colorbar label/tick font scale relative to FONT_LABEL

# [Barb scale factors]
BARB_SCALE_STATION  = 0.80   # barb length scale for continental network plot
VRB_SCATTER_MULTIPLIER          = 1.6  # VRB scatter size multiplier (most charts)
VRB_SCATTER_MULTIPLIER_ENHANCED = 1.4  # VRB scatter size multiplier (enhanced station chart)
BARB_SCALE_UPPER       = 0.65    # barb length scale for gfs analysis upper-air charts

# [850 hPa GPH / Temperature / Wind chart]
GPH_INTERVAL_850      = 3       # geopotential height contour interval (dam)
ISOTHERM_INTERVAL_850 = 1     # temperature fill interval (°C)
TEMP_ADV_INTERVAL_850 = 0.5  # temperature advection fill interval (°C per 1 h)
TEMP_MIN_850          = -30     # lower bound of temperature fill range (°C)
TEMP_MAX_850          = 30      # upper bound of temperature fill range (°C)
TEMP_ADV_MIN_850     = -5      # lower bound of temperature advection fill range (°C per 1 h)
TEMP_ADV_MAX_850     = 5       # upper bound of temperature advection fill range (°C per 1 h)
SIGMA_850             = 3       # Gaussian smoothing sigma (grid points)
GPH_LINEWIDTH_850     = 1.2     # GPH contour line width
GPH_LABEL_STRIDE_850  = 2       # label every Nth GPH contour level
CBAR_TEMP_TICK_850          = 5     # colorbar tick spacing
CBAR_TEMP_ADV_TICK_850          = 1     # colorbar tick spacing

# [Upper-air smoothing and MSLP chart]
SIGMA_UPPER    = 3   # Gaussian smoothing sigma for upper-air GFS fields (grid points)
MSLP_INTERVAL  = 4   # MSLP contour interval for GFS surface chart (hPa)
SIGMA_SURFACE  = 3   # Gaussian smoothing sigma for MSLP
T2M_INTERVAL   = 4   # 2 m temperature contour interval for GFS surface chart (°C)

# [GPH contour intervals and style — by pressure level]
GPH_INTERVAL_700      = 3     # 700 hPa GPH contour interval (dam)
GPH_INTERVAL_500      = 6     # 500 hPa GPH contour interval (dam)
GPH_INTERVAL_250      = 12    # 250 hPa GPH contour interval (dam)
GPH_LINEWIDTH_UPPER   = 1.0   # GPH line width for single-panel upper-air charts
GPH_LINEWIDTH_500     = 1.4   # GPH line width for standalone 500 hPa chart
GPH_LABEL_STRIDE      = 2     # label every Nth GPH level (most charts)
GPH_LABEL_STRIDE_500  = 1     # label every GPH level on standalone 500 hPa chart
GPH_LABEL_STRIDE_250  = 1     # label every GPH level on standalone 250 hPa chart

# [Temperature and RH contour ranges]
TEMP_ISOTHERM_MIN     = -40   # lower bound of surface temperature contour range (°C)
TEMP_ISOTHERM_MAX     = 45    # upper bound (exclusive) of surface temperature contour range (°C)
RH_CONTOUR_LEVELS_700 = [70, 80, 90, 100]  # 700 hPa RH fill boundaries (%) — no fill below 65

# [Jet stream — 250 hPa]
ISOTACH_MIN        = 30       # minimum isotach fill level (m/s)
ISOTACH_MAX        = 100      # exclusive upper bound for isotach fill (m/s)
ISOTACH_INTERVAL   = 2       # isotach fill interval (m/s)
CBAR_ISOTACH_TICK_250 = 10      # isotach colorbar tick spacing (m/s)
JET_CORE_LEVEL     = 50.0     # jet-core emphasis contour level (m/s)
JET_CORE_COLOR     = "darkred"  # jet-core contour color
JET_CORE_LINEWIDTH = 2.0      # jet-core contour line width

# [Colorbar geometry]
CBAR_SIZE         = "2.5%"  # colorbar strip width as fraction of axes width
CBAR_PAD          = 0.2     # colorbar padding from axes edge (inches)

# [Temperature advection and vorticity scaling and color levels]
TEMP_ADV_TIME_SCALE   = 3600 * 1  # K/s → K per 1 h
VORTICITY_DISPLAY_SCALE = 1e5     # vorticity scaling for display (×10⁻⁵ s⁻¹)
VORT_MAX_500          = 40        # half-range for ±symmetric 500 hPa vorticity colorbar (×10⁻⁵)
VORT_INTERVAL_500     = 1         # relative vorticity fill interval (×10⁻⁵ s⁻¹)
CBAR_VORT_TICK_500    = 10        # colorbar tick spacing for 500 hPa vorticity (×10⁻⁵ s⁻¹)

_pvort_cmap = LinearSegmentedColormap.from_list(
    "pvort_div",
    [
        (0.00, "#555555"),
        (0.20, "#999999"),
        (0.38, "#d4d4d4"),
        (0.50, "#f8f5e8"),
        (0.60, "#ffffc0"),
        (0.70, "#ffff00"),
        (0.78, "#ffc200"),
        (0.85, "#ff8c00"),
        (0.91, "#ff3300"),
        (0.95, "#cc0000"),
        (0.98, "#800040"),
        (1.00, "#1a0040"),
    ],
    N=256,
)

# Pre-built 700 hPa RH colormap with three filled bands above 70, 80, 90 %
# Shared by plot_upper_air_overview and plot_four_panel_forecast_animation.
_N_RH    = len(RH_CONTOUR_LEVELS_700) - 1
_RH_CMAP = ListedColormap([plt.cm.BuGn(v) for v in (0.50, 0.70, 0.90)][:_N_RH])
_RH_NORM = BoundaryNorm(RH_CONTOUR_LEVELS_700, ncolors=_N_RH)

# 250 hPa isotach fill levels (m/s), shared by overview and animation.
_ISO_LEVS = list(range(ISOTACH_MIN, ISOTACH_MAX, ISOTACH_INTERVAL))

# [4-panel upper-air overview layout]
FIG_SIZE_SCALE_OVERVIEW      = 1.2    # figure size scale relative to FIG_SIZE_IN
FIG_HEIGHT_OVERVIEW          = 14     # figure height in inches for 4-panel overview (overrides FIG_SIZE_IN[1] × scale)
SUBPLOT_WSPACE_OVERVIEW      = 0.07   # horizontal subplot spacing (2- and 4-panel)
SUBPLOT_HSPACE_OVERVIEW      = 0.02   # vertical subplot spacing (4-panel only)
FONT_SCALE_TITLE_OVERVIEW    = 0.60   # per-panel title font scale relative to FONT_TITLE
FONT_SCALE_CONTOUR_OVERVIEW  = 0.60   # additional contour-label font scale for 4-panel
BARB_STRIDE_SCALE_OVERVIEW   = 2      # barb thinning stride multiplier for 4-panel
BARB_SCALE_OVERVIEW          = 0.70   # barb length scale for 4-panel relative to BARB_SCALE_STATION
FONT_SCALE_SUPTITLE_OVERVIEW = 0.65   # suptitle font scale relative to FONT_TITLE
FIG_DPI_OVERVIEW             = 200    # output DPI for 4-panel overview (lower than single-panel)

# [GFS forecast animation]
_GIF_INTERVAL_MS = 700   # milliseconds per frame in exported GIF
SUBPLOT_HSPACE_OVERVIEW_FORECAST      = 0.09   # vertical subplot spacing (4-panel only)
SUBPLOT_WSPACE_OVERVIEW_FORECAST      = 0.02   # horizontal subplot spacing (2-panel only)

# [500 hPa RAOB station plot]
BARB_SCALE_500_RAOB  = 0.80  # barb length scale for 500 hPa RAOB station plot
RAOB_LABEL_LAT_UPPER = 0.40  # upper label latitude offset (degrees)
RAOB_LABEL_LAT_LOWER = 0.40  # lower label latitude offset (degrees)

# [Label scale factors — Europe map covers ~4× the lat/lon range of Greece]
LABEL_SCALE_STATION = 3.0    # multiplier for lon/lat label offsets on continental network plot

# [Projection — Lambert Conformal for the European domain]
PROJ_CENTRAL_LON    = 15          # central longitude (degrees East)
PROJ_CENTRAL_LAT    = 50          # central latitude  (degrees North)
PROJ_STD_PARALLELS  = (35, 65)    # standard parallels

# [Pressure center detection]
N_SIZE                = 25       # neighbourhood size for extremum filter
SYMBOL_SIZE           = 20       # font size for H/L symbol text
HL_MIN_SEP_DEG        = 25.0     # minimum separation between H/L centers (degrees)
HL_VALUE_LAT_OFFSET   = 0.8      # latitude offset below H/L symbol for pressure value label (degrees)
HL_VALUE_FONT_OFFSET  = 2        # font size reduction for pressure value label relative to symbol
HL_BOUNDARY_MARGIN_DEG = 2.5     # inward margin (degrees) applied to all edges before placing H/L symbols
ISOTHERM_MIN_AREA_KM2 = 200000   # suppress closed isotherms enclosing less than this area (km²)
CLOSED_CONTOUR_ATOL   = 0.05     # coordinate tolerance for closed-contour detection (degrees)

# [GFS grid spacing and derived thinning radius for wind barbs]
_GFS_GRID_DEG    = 0.25
_KM_PER_DEG      = 111.0    # approximate km per degree latitude (barb stride)
_KM_PER_DEG_EQUAT = 111.32  # equatorial km per degree, WGS84 (shoelace area)
_BARB_STRIDE     = max(1, round(THINNING_RADIUS_KM / (_GFS_GRID_DEG * _KM_PER_DEG)))

# [Skew-T figure]
SKEWT_FIG_SIZE          = (9, 12)   # Skew-T figure size (inches)
SKEWT_ROTATION          = 45        # Skew-T temperature axis rotation (degrees)
SKEWT_WIND_STRIDE       = 2         # wind barb thinning stride
SKEWT_PROFILE_LINEWIDTH = 2.0       # linewidth for T and Td profile curves
SKEWT_PARCEL_LINEWIDTH  = 1.5       # linewidth for parcel trajectory
SKEWT_CAPE_ALPHA        = 0.25      # alpha for CAPE shading
SKEWT_CIN_ALPHA         = 0.25      # alpha for CIN shading
SKEWT_ABOVE_EL_ALPHA    = 0.15      # alpha for negative area above EL shading
SKEWT_MARKER_COLOR      = "k"       # color for LCL/LFC/EL horizontal markers
SKEWT_MARKER_LINEWIDTH  = 0.8       # linewidth for LCL/LFC/EL markers
SKEWT_MARKER_ALPHA      = 0.6       # alpha for LCL/LFC/EL markers
SKEWT_MARKER_XMIN       = 0.88      # left extent of LCL/LFC/EL marker lines (axes fraction)
SKEWT_MARKER_XMAX       = 1.0       # right extent of LCL/LFC/EL marker lines

# [Skew-T convective-level tick markers]
SKEWT_LEVEL_LINE_X_START  = 0.85
SKEWT_LEVEL_LINE_X_END    = 0.92
SKEWT_LEVEL_LINE_WIDTH    = 1.8
SKEWT_LEVEL_LINE_ALPHA    = 0.85
SKEWT_LEVEL_LABEL_X       = 0.91
SKEWT_LEVEL_LABEL_FONTSIZE = 9

# [Skew-T instability annotation box]
SKEWT_ANNOT_X             = 0.02
SKEWT_ANNOT_Y             = 0.03
SKEWT_ANNOT_FONTSIZE      = 10
SKEWT_ANNOT_LINESPACING   = 1.6
SKEWT_ANNOT_ALPHA         = 0.85

# [Skew-T reference lines]
SKEWT_ISOBAR_LEVELS          = [100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SKEWT_ISOBAR_COLOR           = "#888888"    # isobar color
SKEWT_ISOBAR_LINEWIDTH       = 0.7          # isobar line width
SKEWT_ISOBAR_ALPHA           = 0.8          # isobar alpha
SKEWT_ISOTHERM_MIN           = -70          # leftmost isotherm (°C)
SKEWT_ISOTHERM_MAX           = 60           # rightmost isotherm (°C)
SKEWT_ISOTHERM_INTERVAL      = 10           # isotherm spacing (°C)
SKEWT_ISOTHERM_COLOR         = "#888888"    # isotherm color
SKEWT_ISOTHERM_LINEWIDTH     = 0.5          # isotherm line width
SKEWT_ISOTHERM_ALPHA         = 0.6          # isotherm alpha
SKEWT_ZERO_ISOTHERM_COLOR    = "#3a7ebf"  # 0 °C isotherm color
SKEWT_ZERO_ISOTHERM_STYLE    = "--"      # 0 °C isotherm style
SKEWT_ZERO_ISOTHERM_LINEWIDTH = 1.2         # 0 °C isotherm line width
SKEWT_ZERO_ISOTHERM_ALPHA    = 0.8          # 0 °C isotherm alpha
SKEWT_MIXING_RATIO_LEVELS    = [0.4, 1, 2, 4, 7, 10, 16, 24, 32]  # (g/kg)
SKEWT_MIXING_LINE_STYLE      = "--"         # mixing ratio line style
SKEWT_MIXING_LINE_COLOR      = "#AADBAA"    # mixing ratio line color
SKEWT_MIXING_LINE_WIDTH      = 1.2          # mixing ratio line width
SKEWT_MIXING_LINE_ALPHA      = 0.5          # mixing ratio line alpha
SKEWT_MIXING_LABEL_FONTSIZE  = 9            # font size for mixing ratio line labels
SKEWT_DRY_ADIABAT_INTERVAL   = 10           # dry adiabat interval (°C)
SKEWT_DRY_ADIABAT_STYLE      = "--"         # dry adiabat line style
SKEWT_DRY_ADIABAT_COLOR      = "#E59F9F"    # dry adiabat color
SKEWT_DRY_ADIABAT_LINEWIDTH  = 1.2          # dry adiabat line width
SKEWT_DRY_ADIABAT_ALPHA      = 0.5          # dry adiabat alpha
SKEWT_MOIST_ADIABAT_INTERVAL = 5            # moist adiabat interval (°C)
SKEWT_MOIST_ADIABAT_COLOR    = "#A3C4E6"    # moist adiabat color
SKEWT_MOIST_ADIABAT_STYLE    = "--"         # moist adiabat line style
SKEWT_MOIST_ADIABAT_LINEWIDTH = 1.2         # moist adiabat line width
SKEWT_MOIST_ADIABAT_ALPHA    = 0.5          # moist adiabat alpha

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("owl.plot.helpers")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


# [Internal helpers]

def _suppress_small_closed_contours(
    cs,
    ax,
    data_crs,
    children_before: set,
    color: str,
    linewidth: float,
    threshold_km2: float = ISOTHERM_MIN_AREA_KM2,
) -> None:
    """
    Suppress closed isotherms smaller than *threshold_km2*.

    Strategy: hide all GeoLineCollection artists added by *cs*, then redraw
    only the kept segments as a plain ``LineCollection`` built from
    ``cs.allsegs`` — raw (lon, lat) arrays that are always accessible
    regardless of matplotlib/cartopy version.

    Parameters
    ----------
    cs : cartopy GeoContourSet
    ax : cartopy GeoAxes
    data_crs : cartopy CRS  (PlateCarree)
    children_before : set of int
        ``{id(c) for c in ax.get_children()}`` snapshot taken **before**
        calling ``ax.contour()``.
    color, linewidth : style to apply to the redrawn segments.
    threshold_km2 : float
        Closed segments enclosing less than this area are dropped.
        Default: ``ISOTHERM_MIN_AREA_KM2``.
    """
    # Hide every artist added by the contour call
    for child in ax.get_children():
        if id(child) not in children_before and isinstance(child, mcoll.LineCollection):
            child.set_visible(False)

    # Rebuild from cs.allsegs — one list of (N,2) arrays per level, in data CRS
    kept_segs = []
    for level_segs in cs.allsegs:
        for seg in level_segs:
            if len(seg) < 3:
                kept_segs.append(seg)
                continue
            is_closed = np.allclose(seg[0], seg[-1], atol=CLOSED_CONTOUR_ATOL)
            if not is_closed:
                kept_segs.append(seg)
                continue
            # Area via shoelace on km coordinates
            seg_lon, seg_lat = seg[:, 0], seg[:, 1]
            mean_lat = float(np.mean(seg_lat))
            kx = seg_lon * _KM_PER_DEG_EQUAT * np.cos(np.radians(mean_lat))
            ky = seg_lat * _KM_PER_DEG_EQUAT
            area_km2 = 0.5 * abs(
                np.dot(kx, np.roll(ky, -1)) - np.dot(ky, np.roll(kx, -1))
            )
            if area_km2 >= threshold_km2:
                kept_segs.append(seg)

    if kept_segs:
        lc = mcoll.LineCollection(
            kept_segs, colors=color, linewidths=linewidth, transform=data_crs,
        )
        ax.add_collection(lc)


def _setup_europe_map(ax=None, proj=None):
    """
    Create or configure a Europe-domain map.

    If *ax* is None a new figure and axes are created; otherwise the
    existing axes are configured in place.

    Parameters
    ----------
    proj : cartopy.crs.Projection, optional
        Map projection. Defaults to Lambert Conformal centred at
        PROJ_CENTRAL_LON / PROJ_CENTRAL_LAT.
    
    Returns
    -------
    tuple (fig, ax, proj, data_crs)
    """
    if proj is None:
        proj = ccrs.LambertConformal(
            central_longitude=PROJ_CENTRAL_LON,
            central_latitude=PROJ_CENTRAL_LAT,
            standard_parallels=PROJ_STD_PARALLELS,
        )
    data_crs = ccrs.PlateCarree()

    if ax is None:
        fig = plt.figure(figsize=FIG_SIZE_IN)
        fig.patch.set_facecolor("w")
        ax = plt.axes(projection=proj)
    else:
        fig = ax.figure

    ax.set_extent(EUROPE_EXTENT, crs=data_crs)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.9)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray", alpha=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

    return fig, ax, proj, data_crs


def _smooth_field(data_2d: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply NaN-safe Gaussian smoothing to a 2-D field.
    """
    nan_mask = np.isnan(data_2d)
    filled   = data_2d.copy().astype(float)
    if nan_mask.any():
        filled[nan_mask] = float(np.nanmean(data_2d))
    smoothed = gaussian_filter(filled, sigma=sigma)
    smoothed[nan_mask] = np.nan
    return smoothed


def _gfs_title(field_label: str, valid_time: str) -> str:
    """
    Format a consistent GFS-analysis map title.
    """
    vt = pd.Timestamp(valid_time, tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    return f"{field_label}\nGFS analysis valid: {vt}"


def _gefs_title(field_label: str, init_date: str, lead_time_h: int) -> str:
    """
    Format a consistent GEFS ensemble map title.
    """
    init_ts  = pd.Timestamp(init_date)
    valid_ts = init_ts + pd.Timedelta(hours=lead_time_h)
    return (
        f"{field_label}\n"
        f"GEFS +{lead_time_h} h | Init: {init_ts.strftime('%Y-%m-%d %H:%M')} UTC | "
        f"Valid: {valid_ts.strftime('%Y-%m-%d %H:%M')} UTC"
    )


def _add_skewt_annot_box(skew, lines: list[str]) -> None:
    """
    Place a white rounded annotation box in the upper-left corner of the Skew-T axes.
    """
    skew.ax.text(
        SKEWT_ANNOT_X, 1 - SKEWT_ANNOT_X,
        "\n".join(lines),
        transform=skew.ax.transAxes,
        fontsize=SKEWT_ANNOT_FONTSIZE, va="top", ha="left",
        linespacing=SKEWT_ANNOT_LINESPACING,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=SKEWT_ANNOT_ALPHA),
        zorder=6,
    )

# [Loaders]

def fetch_station_coords(
    url: str = AIRPORTS_URL,
    iso_country: str = ISO_COUNTRY,
) -> pd.DataFrame:
    """
    Fetch ICAO station coordinates from the OurAirports open database.

    Parameters
    ----------
    url : str
        URL of the OurAirports airports.csv file.
    iso_country : str
        Two-letter ISO country code used to filter the airport list
        (default 'GR' for Greece).

    Returns
    -------
    pd.DataFrame
        Columns: ``station`` (ICAO ident), ``lat`` (degrees N),
        ``lon`` (degrees E).
        Only airports whose ``ident`` starts with the expected ICAO prefix
        (first two letters of the country's ICAO block) are retained.
    """
    airports = pd.read_csv(
        url,
        usecols=["ident", "latitude_deg", "longitude_deg", "iso_country"],
        dtype={"ident": str, "iso_country": str},
    )

    # Filter to target country
    airports = airports[airports["iso_country"] == iso_country].copy()

    # Keep only proper 4-letter ICAO identifiers
    airports = airports[airports["ident"].str.len() == 4]

    airports = airports.rename(columns={
        "ident":         "station",
        "latitude_deg":  "lat",
        "longitude_deg": "lon",
    })

    return airports[["station", "lat", "lon"]].reset_index(drop=True)

def fetch_ourairports_europe(
    lat_min: float = 30,
    lat_max: float = 72,
    lon_min: float = -25,
    lon_max: float = 45,
    url: str = OURAIRPORTS_FULL_URL,
) -> pd.DataFrame:
    """
    Fetch European airport metadata from the OurAirports open database.

    Parameters
    ----------
    lat_min : float
        Southern boundary of the bounding box (degrees North).
    lat_max : float
        Northern boundary of the bounding box (degrees North).
    lon_min : float
        Western boundary of the bounding box (degrees East).
    lon_max : float
        Eastern boundary of the bounding box (degrees East).
    url : str
        URL of the OurAirports airports.csv file.

    Returns
    -------
    pd.DataFrame
        Columns:

        ============== =============================================
        icao_code      4-letter ICAO identifier (e.g. ``'EGLL'``)
        name           Airport name (e.g. ``'London Heathrow Airport'``)
        latitude_deg   Latitude in degrees North
        longitude_deg  Longitude in degrees East
        elevation_m    Elevation in metres above mean sea level
        ============== =============================================

        Only airports with ``type`` in ``['large_airport',
        'medium_airport', 'small_airport']``, non-null ``ident``,
        non-null coordinates, and within the specified bounding box
        are returned.
    """
    airports = pd.read_csv(
        url,
        usecols=[
            "ident", "type", "name",
            "latitude_deg", "longitude_deg", "elevation_ft",
        ],
        dtype={"ident": str, "type": str, "name": str},
    )

    # Filter by airport type.  
    airports = airports[airports["type"].isin(VALID_AIRPORT_TYPES)].copy()

    # Require a valid 4-letter ICAO code and non-null coordinates
    airports = airports.dropna(subset=["ident", "latitude_deg", "longitude_deg"])
    airports = airports[airports["ident"].str.len() == 4]

    # Subset to the European bounding box
    airports = airports[
        (airports["latitude_deg"]  >= lat_min) &
        (airports["latitude_deg"]  <= lat_max) &
        (airports["longitude_deg"] >= lon_min) &
        (airports["longitude_deg"] <= lon_max)
    ].copy()

    # Rename ident -> icao_code for clarity; convert elevation to metres
    airports = airports.rename(columns={"ident": "icao_code"})
    airports["elevation_m"] = (airports["elevation_ft"] * 0.3048).round(1)

    return airports[
        ["icao_code", "name", "latitude_deg", "longitude_deg", "elevation_m"]
    ].reset_index(drop=True)

# [Core logic]

def build_network_plot_df(
    df_clean: pd.DataFrame,
    coords: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one representative observation per station and merge with coords.

    Selection logic:
    - Real-time mode  (1 row per station): use that row directly.
    - Retrospective   (multiple rows):     pick the observation closest to
                                           12:00 UTC (by minutes-from-noon).

    Parameters
    ----------
    df_clean : pd.DataFrame
        QC-passed observation table produced by previous notebook cells.
        Must contain a ``station`` column and a ``valid`` datetime column.
    coords : pd.DataFrame
        Station coordinate table from ``fetch_station_coords()``.

    Returns
    -------
    pd.DataFrame
        One row per station, merged with ``lat``/``lon`` from the coordinate
        table. Stations without a matching coordinate are dropped silently.
    """
    rows: list[pd.Series] = []

    for stn, grp in df_clean.groupby("station"):
        grp = grp.sort_values("valid")

        if len(grp) == 1:
            rows.append(grp.iloc[0])
        else:
            grp = grp.copy()
            grp["_mfn"] = (
                grp["valid"].dt.hour * 60 + grp["valid"].dt.minute - 720
            ).abs()
            rows.append(grp.loc[grp["_mfn"].idxmin()])

    if not rows:
        return pd.DataFrame()

    df_rep = pd.DataFrame(rows).reset_index(drop=True)
    df_plot = df_rep.merge(coords, on="station", how="inner")
    return df_plot

def build_europe_network_plot_df(
    df_clean: pd.DataFrame,
    station_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one representative observation per station and merge with European
    station coordinates from OurAirports.

    Selection logic is identical to ``build_network_plot_df()``:

    - Real-time mode  (1 row per station): use that row directly.
    - Retrospective   (multiple rows):     pick the observation closest to
                                           12:00 UTC (by minutes-from-noon).


    Parameters
    ----------
    df_clean : pd.DataFrame
        QC-passed observation table.  Must contain ``station`` and ``valid``
        columns.
    station_meta : pd.DataFrame
        OurAirports metadata from ``fetch_ourairports_europe()``.  Must
        contain ``icao_code``, ``latitude_deg``, ``longitude_deg``.

    Returns
    -------
    pd.DataFrame
        One row per station, with ``lat`` and ``lon`` columns
        added from *station_meta*. Stations without matching metadata are dropped silently.
    """
    rows: list[pd.Series] = []

    for stn, grp in df_clean.groupby("station"):
        grp = grp.sort_values("valid")

        if len(grp) == 1:
            rows.append(grp.iloc[0])
        else:
            grp = grp.copy()
            grp["_mfn"] = (
                grp["valid"].dt.hour * 60 + grp["valid"].dt.minute - 720
            ).abs()
            rows.append(grp.loc[grp["_mfn"].idxmin()])

    if not rows:
        return pd.DataFrame()

    df_rep = pd.DataFrame(rows).reset_index(drop=True)

    # Merge with station metadata (icao_code -> station)
    meta = station_meta.rename(columns={
        "icao_code":     "station",
        "latitude_deg":  "lat",
        "longitude_deg": "lon",
    })

    n_before = len(df_rep)
    df_plot = df_rep.merge(
        meta[["station", "lat", "lon"]],
        on="station",
        how="inner",
    )
    n_dropped = n_before - len(df_plot)

    if n_dropped > 0:
        LOG.warning(
            "%d station(s) in df_clean not found in station_meta — skipped.",
            n_dropped,
        )

    return df_plot


# [Pressure center detection]

def plot_maxmin_points(
    ax,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    data: np.ndarray,
    extrema: str,
    n_size: int = N_SIZE,
    symbol_size: int = SYMBOL_SIZE,
    min_sep_deg: float = HL_MIN_SEP_DEG,
    transform=None,
) -> None:
    """
    Plot H/L symbols at local pressure maxima or minima.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Cartopy GeoAxes.
    lon_grid : np.ndarray
        1-D longitude array (ascending).
    lat_grid : np.ndarray
        1-D latitude array (ascending).
    data : np.ndarray
        2-D array of shape ``(len(lat_grid), len(lon_grid))`` — typically
        smoothed MSLP.
    extrema : str
        ``'max'`` for high-pressure centers, ``'min'`` for low-pressure
        centers.
    n_size : int
        Neighbourhood size for the local extremum filter (larger = fewer,
        more synoptic-scale centers).
    symbol_size : int
        Font size for the H/L symbol text.
    transform : cartopy.crs.Projection, optional
        CRS of *lon_grid* / *lat_grid*. Defaults to ``ccrs.PlateCarree()``.

    Returns
    -------
    None
    """
    if transform is None:
        transform = ccrs.PlateCarree()

    # Determine filter function, label, and colour
    if extrema == "max":
        data_ext = maximum_filter(data, n_size, mode="nearest")
        label, color = "H", "blue"
    elif extrema == "min":
        data_ext = minimum_filter(data, n_size, mode="nearest")
        label, color = "L", "red"
    else:
        raise ValueError(f"extrema must be 'max' or 'min', got '{extrema}'")

    # Get current map extent to skip off-screen centers
    extent = ax.get_extent(crs=transform)
    lon_min_e, lon_max_e, lat_min_e, lat_max_e = extent

    # Build 2-D coordinate arrays
    mesh_lon, mesh_lat = np.meshgrid(lon_grid, lat_grid)

    # Find locations where data equals the filtered extreme
    # (and data is not NaN)
    matches = (data == data_ext) & ~np.isnan(data)
    match_locs = np.argwhere(matches)

    # Minimum-separation thinning: keep only the most extreme center
    # when two detections are closer than min_sep_deg degrees
    if min_sep_deg > 0 and len(match_locs) > 1:
        lons_all = np.array([mesh_lon[iy, ix] for iy, ix in match_locs])
        lats_all = np.array([mesh_lat[iy, ix] for iy, ix in match_locs])
        vals_all = np.array([data[iy, ix] for iy, ix in match_locs])
        order = np.argsort(vals_all)[::-1] if extrema == "max" else np.argsort(vals_all)
        kept = []
        for idx in order:
            too_close = any(
                np.sqrt((lons_all[idx] - lons_all[k]) ** 2 +
                        (lats_all[idx] - lats_all[k]) ** 2) < min_sep_deg
                for k in kept
            )
            if not too_close:
                kept.append(idx)
        match_locs = [match_locs[i] for i in kept]

    for iy, ix in match_locs:
        lon_val = mesh_lon[iy, ix]
        lat_val = mesh_lat[iy, ix]

        # Skip if outside current map extent
        margin_deg = HL_BOUNDARY_MARGIN_DEG
        if not (lon_min_e + margin_deg <= lon_val <= lon_max_e - margin_deg and
                lat_min_e + margin_deg <= lat_val <= lat_max_e - margin_deg):
            continue

        val = data[iy, ix]

        ax.text(
            lon_val, lat_val, label,
            color=color, fontsize=symbol_size, fontweight="bold",
            ha="center", va="center",
            transform=transform, zorder=10,
        )
        ax.text(
            lon_val, lat_val - HL_VALUE_LAT_OFFSET, f"{val:.0f}",
            color=color, fontsize=symbol_size - HL_VALUE_FONT_OFFSET,
            ha="center", va="top",
            transform=transform, zorder=10,
        )

# [Surface maps]

def plot_greece_metar_network(
    df_plot: pd.DataFrame,
    output_dir: str = "../outputs",
    title_suffix: Optional[str] = None,
) -> str:
    """
    Draw simplified station plots for the Greece METAR network.

    For each station the following elements are drawn:

    - **Wind barbs** (knots, black) for stations with a known direction
    - **VRB scatter** (black dot, size proportional to speed) for variable-
      direction wind
    - **Calm circle** (open black circle) for wind speed <= CALM_THRESHOLD_KT kt
    - **Temperature** (degC, red text, upper-left of position)
    - **Relative Humidity** (%, blue text, lower-left of position)

    Parameters
    ----------
    df_plot : pd.DataFrame
        Output of ``build_network_plot_df()``.  Required columns:
        ``station``, ``lat``, ``lon``, ``temp_c``, ``relh``,
        ``u_kt``, ``v_kt``, ``wspd``, ``wdir``.
    output_dir : str
        Directory where the PNG is saved.  Created if it does not exist.
    title_suffix : str, optional
        Extra text appended to the map title (e.g., a date string).

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    if df_plot.empty:
        raise ValueError("df_plot is empty — nothing to plot.")

    proj = ccrs.PlateCarree()

    # ── Classify observations by wind type ────────────────────────────────
    # Calm  : valid direction, speed <= CALM_THRESHOLD_KT
    # VRB   : wdir is NaN (variable direction)
    # Barb  : everything else (directional, speed > CALM_THRESHOLD_KT)
    df = df_plot.copy()

    is_vrb  = df["wdir"].isna()
    is_calm = (~is_vrb) & (df["wspd"].fillna(0) <= CALM_THRESHOLD_KT)
    is_barb = (~is_vrb) & (~is_calm)

    df_barb = df[is_barb]
    df_vrb  = df[is_vrb]
    df_calm = df[is_calm]

    # ── Map canvas ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=FIG_SIZE_IN)
    fig.patch.set_facecolor("w")
    ax = plt.axes(projection=proj)
    ax.set_extent(GREECE_EXTENT, crs=proj)

    ax.add_feature(cfeature.COASTLINE.with_scale("10m"))
    ax.add_feature(cfeature.BORDERS.with_scale("10m"))
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="lightgray")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray",
                 alpha=0.4, linestyle="--")

    # ── Wind ──────────────────────────────────────────────────────────────
    if not df_barb.empty:
        ax.barbs(
            df_barb["lon"], df_barb["lat"],
            df_barb["u_kt"], df_barb["v_kt"],
            length=BARB_LENGTH, transform=proj, color="black",
        )

    if not df_vrb.empty:
        ax.scatter(
            df_vrb["lon"], df_vrb["lat"],
            s=np.clip(df_vrb["wspd"] * VRB_SCATTER_MULTIPLIER, VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
            c="black", transform=proj, label="VRB",
        )

    if not df_calm.empty:
        ax.scatter(
            df_calm["lon"], df_calm["lat"],
            s=CALM_CIRCLE_S, facecolors="none", edgecolors="black",
            transform=proj, label="Calm",
        )

    # ── Labels: T (red, upper-left) + RH (blue, lower-left) ─────────────
    for _, rr in df.iterrows():
        if pd.notna(rr.get("temp_c", np.nan)):
            ax.text(
                rr["lon"] - LABEL_LON_OFFSET, rr["lat"] + LABEL_LAT_UPPER,
                f"{rr['temp_c']:.0f} \u00b0C",
                color="red", fontsize=FONT_LABEL, ha="right",
                transform=proj, clip_on=True,
            )
        if pd.notna(rr.get("relh", np.nan)):
            ax.text(
                rr["lon"] - LABEL_LON_OFFSET, rr["lat"] - LABEL_LAT_LOWER,
                f"{rr['relh']:.0f}%",
                color="blue", fontsize=FONT_LABEL, ha="right",
                transform=proj, clip_on=True,
            )
    # ── Title and station count ───────────────────────────────────────────
    obs_times = pd.to_datetime(df["valid"]).dropna()
    if len(obs_times) > 0:
        t_min = obs_times.min()
        t_max = obs_times.max()
        if t_min.strftime("%Y-%m-%d %H") == t_max.strftime("%Y-%m-%d %H"):
            time_label = f"Latest METAR: {t_max.strftime('%Y-%m-%d %H:%M UTC')}"
        else:
            time_label = (f"METAR range: {t_min.strftime('%Y-%m-%d %H:%M')} "
                          f"\u2013 {t_max.strftime('%H:%M')} UTC")
    else:
        time_label = "Latest METAR: unknown time"

    title_line = f"Surface Weather Map\n{time_label}"
    if title_suffix:
        title_line += f"  |  {title_suffix}"

    ax.set_title(title_line, fontsize=FONT_TITLE, loc="left")
    plt.text(
        0.98, 1.02, f"{len(df)} stations",
        ha="right", va="bottom", fontsize=FONT_COUNT, transform=ax.transAxes,
    )

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"greece_metar_network_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()

    return str(out_path.resolve())


def plot_europe_metar_network(
    df_plot: pd.DataFrame,
    output_dir: str = "../outputs",
    title_suffix: Optional[str] = None,
) -> str:
    """
    Draw simplified station plots for the European METAR network.

    For each station the following elements are drawn:

    - **Wind barbs** (knots, black) for stations with a known direction
    - **VRB scatter** (black dot, size proportional to speed) for
      variable-direction wind
    - **Calm circle** (open black circle) for wind speed <= 1 kt
    - **Temperature** (degC, red text, upper-left of position)
    - **Relative Humidity** (%, blue text, lower-left of position)

    Parameters
    ----------
    df_plot : pd.DataFrame
        Output of ``build_europe_network_plot_df()``.  Required columns:
        ``station``, ``lat``, ``lon``, ``temp_c``, ``relh``,
        ``u_kt``, ``v_kt``, ``wspd``, ``wdir``.
    output_dir : str
        Directory where the PNG is saved.  Created if it does not exist.
    title_suffix : str, optional
        Extra text appended to the map title (e.g., a date string).

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    if df_plot.empty:
        raise ValueError("df_plot is empty — nothing to plot.")

    proj = ccrs.LambertConformal(
        central_longitude=PROJ_CENTRAL_LON,
        central_latitude=PROJ_CENTRAL_LAT,
        standard_parallels=PROJ_STD_PARALLELS,
    )
    data_crs = ccrs.PlateCarree()

    # ── Thin and classify observations by wind type ───────────────────────
    # Calm  : valid direction, speed <= CALM_THRESHOLD_KT
    # VRB   : wdir is NaN (variable direction)
    # Barb  : everything else (directional, speed > CALM_THRESHOLD_KT)
    if THINNING_RADIUS_KM > 0:
        proj_points = proj.transform_points(data_crs, df_plot["lon"].values, df_plot["lat"].values)
        mask = reduce_point_density(proj_points, THINNING_RADIUS_KM * 1000)
        df = df_plot[mask].copy()
    else:
        df = df_plot.copy()

    is_vrb  = df["wdir"].isna()
    is_calm = (~is_vrb) & (df["wspd"].fillna(0) <= CALM_THRESHOLD_KT)
    is_barb = (~is_vrb) & (~is_calm)

    df_barb = df[is_barb]
    df_vrb  = df[is_vrb]
    df_calm = df[is_calm]

    # ── Map canvas ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=FIG_SIZE_IN)
    fig.patch.set_facecolor("w")
    ax = plt.axes(projection=proj)
    ax.set_extent(EUROPE_EXTENT, crs=data_crs)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray",
                   alpha=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray",
                 alpha=0.4, linestyle="--")

    # ── Wind ──────────────────────────────────────────────────────────────
    if not df_barb.empty:
        ax.barbs(
            df_barb["lon"], df_barb["lat"],
            df_barb["u_kt"], df_barb["v_kt"],
            length=BARB_LENGTH * BARB_SCALE_STATION, transform=data_crs, color="black",
        )

    if not df_vrb.empty:
        ax.scatter(
            df_vrb["lon"], df_vrb["lat"],
            s=np.clip(df_vrb["wspd"] * 1.6,
                      VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
            c="black", transform=data_crs, label="VRB",
        )

    if not df_calm.empty:
        ax.scatter(
            df_calm["lon"], df_calm["lat"],
            s=CALM_CIRCLE_S, facecolors="none", edgecolors="black",
            transform=data_crs, label="Calm",
        )

    # ── Labels: T + RH (left) | P (top-right) ────────────────────────────
    for _, rr in df.iterrows():
        _fs = max(1, round(FONT_LABEL * FONT_SCALE_STATION))
        if pd.notna(rr.get("temp_c", np.nan)):
            ax.text(
                rr["lon"] - LABEL_LON_OFFSET * LABEL_SCALE_STATION,
                rr["lat"] + LABEL_LAT_UPPER * LABEL_SCALE_STATION,
                f"{rr['temp_c']:.0f} °C",
                color="red", fontsize=_fs, ha="right",
                transform=data_crs, clip_on=True,
            )
        if pd.notna(rr.get("relh", np.nan)):
            ax.text(
                rr["lon"] - LABEL_LON_OFFSET * LABEL_SCALE_STATION,
                rr["lat"] - LABEL_LAT_LOWER * LABEL_SCALE_STATION,
                f"{rr['relh']:.0f}%",
                color="blue", fontsize=_fs, ha="right",
                transform=data_crs, clip_on=True,
            )
        if pd.notna(rr.get("mslp", np.nan)):
            ax.text(
                rr["lon"] + LABEL_LON_OFFSET * LABEL_SCALE_STATION,
                rr["lat"] + LABEL_LAT_UPPER * LABEL_SCALE_STATION,
                f"{round(rr['mslp'] * 10) % 1000:03d}",
                color="black", fontsize=_fs, ha="left",
                transform=data_crs, clip_on=True,
            )

    # ── Title and station count ───────────────────────────────────────────
    obs_times = pd.to_datetime(df["valid"]).dropna()
    if len(obs_times) > 0:
        t_min = obs_times.min()
        t_max = obs_times.max()
        if t_min.strftime("%Y-%m-%d %H") == t_max.strftime("%Y-%m-%d %H"):
            time_label = f"Latest METAR: {t_max.strftime('%Y-%m-%d %H:%M UTC')}"
        else:
            time_label = (f"METAR range: {t_min.strftime('%Y-%m-%d %H:%M')} "
                          f"\u2013 {t_max.strftime('%H:%M')} UTC")
    else:
        time_label = "Latest METAR: unknown time"

    title_line = f"Surface Weather Map\n{time_label}"
    if title_suffix:
        title_line += f"  |  {title_suffix}"

    ax.set_title(title_line, fontsize=FONT_TITLE, loc="left", fontweight="bold")
    plt.text(
        0.98, 1.02, f"{len(df)} stations",
        ha="right", va="bottom", fontsize=FONT_COUNT,
        transform=ax.transAxes,
    )

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"europe_metar_network_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()

    return str(out_path.resolve())


def plot_europe_mslp_raw(
    df_geo: pd.DataFrame,
    output_dir: str = "../outputs",
) -> str:
    """
    Draw a map with raw Mean Sea-Level Pressure (MSLP) values at station locations.

    The observation time is derived automatically from the ``valid`` column
    of *df_geo* and included in the map title.

    Parameters
    ----------
    df_geo : pd.DataFrame
        DataFrame with coordinates and ``mslp`` column; must contain a
        ``valid`` datetime column for the automatic time label.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    if df_geo.empty:
        raise ValueError("df_geo is empty -- nothing to plot.")

    fig, ax, proj, data_crs = _setup_europe_map()

    # Plot MSLP values as text
    for _, row in df_geo.iterrows():
        if pd.notna(row.get("mslp", float("nan"))):
            ax.text(
                row["lon"], row["lat"], f"{row['mslp']:.0f}",
                transform=data_crs, fontsize=max(1, round(FONT_LABEL * FONT_SCALE_MMSLP_RAW)), ha="center", va="center",
                color="darkblue", clip_on=True,
            )

    # Time label
    obs_times = pd.to_datetime(df_geo["valid"]).dropna()
    if len(obs_times) > 0:
        t_min, t_max = obs_times.min(), obs_times.max()
        if t_min.strftime("%Y-%m-%d %H") == t_max.strftime("%Y-%m-%d %H"):
            time_label = f"Latest METAR: {t_max.strftime('%Y-%m-%d %H:%M UTC')}"
        else:
            time_label = (f"METAR range: {t_min.strftime('%Y-%m-%d %H:%M')} "
                          f"\u2013 {t_max.strftime('%H:%M')} UTC")
    else:
        time_label = "Latest METAR: unknown time"

    title_line = f"Station MSLP (hPa)\n{time_label}"
    ax.set_title(title_line, fontsize=FONT_TITLE, loc="left", fontweight="bold")
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"europe_slp_raw_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.show()

    return str(out_path.resolve())


def plot_europe_isobars(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    mslp_grid: np.ndarray,
    output_dir: str = "../outputs",
    isobar_interval: int = 4,
    df_geo: Optional[pd.DataFrame] = None,
) -> str:
    """
    Draw a map with only Mean Sea-Level Pressure computed isobars.

    The observation time is derived automatically from the ``valid`` column
    of *df_geo* when provided; otherwise the current UTC time is used.

    Parameters
    ----------
    grid_lon : np.ndarray
        1-D array of grid longitudes (degrees East).
    grid_lat : np.ndarray
        1-D array of grid latitudes (degrees North).
    mslp_grid : np.ndarray
        2-D array of smoothed MSLP values (hPa), shape
        ``(len(grid_lat), len(grid_lon))``.
    output_dir : str
        Directory to save the figure.
    isobar_interval : int
        Contour interval in hPa. Default 4 hPa (WMO standard).
    df_geo : pd.DataFrame, optional
        Observation DataFrame containing a ``valid`` datetime column used to
        derive the time label in the map title. When omitted, the current UTC
        time is used as a fallback.

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    # ── Derive time label ─────────────────────────────────────────────────
    if df_geo is not None and not df_geo.empty:
        obs_times = pd.to_datetime(df_geo["valid"]).dropna()
        if len(obs_times) > 0:
            t_min, t_max = obs_times.min(), obs_times.max()
            if t_min.strftime("%Y-%m-%d %H") == t_max.strftime("%Y-%m-%d %H"):
                time_label = f"Latest METAR: {t_max.strftime('%Y-%m-%d %H:%M UTC')}"
            else:
                time_label = (f"METAR range: {t_min.strftime('%Y-%m-%d %H:%M')} "
                              f"\u2013 {t_max.strftime('%H:%M')} UTC")
        else:
            time_label = "Latest METAR: unknown time"
    else:
        time_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    fig, ax, proj, data_crs = _setup_europe_map()

    isobar_levels = np.arange(900, 1060, isobar_interval)

    cs = ax.contour(
        grid_lon, grid_lat, mslp_grid,
        levels=isobar_levels, colors="black", linewidths=1.2,
        transform=data_crs,
    )
    ax.clabel(cs, cs.levels[::1], fmt="%d", fontsize=max(1, round(FONT_LABEL)), inline=True)

    title_line = f"Mean Sea-Level Pressure Isobars ({isobar_interval} hPa interval)\n{time_label}"
    ax.set_title(title_line, fontsize=FONT_TITLE, loc="left", fontweight="bold")
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"europe_isobars_only_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.show()

    return str(out_path.resolve())


def plot_europe_isobars_hl(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    mslp_grid: np.ndarray,
    output_dir: str = "../outputs",
    isobar_interval: int = 4,
    df_geo: Optional[pd.DataFrame] = None,
) -> str:
    """
    Draw a map with Mean Sea-Level Pressure isobars and annotated H/L pressure centers.

    The observation time is derived automatically from the ``valid`` column
    of *df_geo* when provided; otherwise the current UTC time is used.

    Parameters
    ----------
    grid_lon : np.ndarray
        1-D array of grid longitudes (degrees East).
    grid_lat : np.ndarray
        1-D array of grid latitudes (degrees North).
    mslp_grid : np.ndarray
        2-D array of smoothed MSLP values (hPa), shape
        ``(len(grid_lat), len(grid_lon))``.
    output_dir : str
        Directory to save the figure.
    isobar_interval : int
        Contour interval in hPa. Default 4 hPa (WMO standard).
    df_geo : pd.DataFrame, optional
        Observation DataFrame containing a ``valid`` datetime column used to
        derive the time label in the map title. When omitted, the current UTC
        time is used as a fallback.

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    # ── Derive time label ─────────────────────────────────────────────────
    if df_geo is not None and not df_geo.empty:
        obs_times = pd.to_datetime(df_geo["valid"]).dropna()
        if len(obs_times) > 0:
            t_min, t_max = obs_times.min(), obs_times.max()
            if t_min.strftime("%Y-%m-%d %H") == t_max.strftime("%Y-%m-%d %H"):
                time_label = f"Latest METAR: {t_max.strftime('%Y-%m-%d %H:%M UTC')}"
            else:
                time_label = (f"METAR range: {t_min.strftime('%Y-%m-%d %H:%M')} "
                              f"\u2013 {t_max.strftime('%H:%M')} UTC")
        else:
            time_label = "Latest METAR: unknown time"
    else:
        time_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    fig, ax, proj, data_crs = _setup_europe_map()

    isobar_levels = np.arange(900, 1060, isobar_interval)

    # Isobars
    cs = ax.contour(
        grid_lon, grid_lat, mslp_grid,
        levels=isobar_levels, colors="black", linewidths=1.2,
        transform=data_crs,
    )
    ax.clabel(cs, cs.levels[::2], fmt="%d", fontsize=max(1, round(FONT_LABEL)), inline=True)

    # H/L centers
    plot_maxmin_points(ax, grid_lon, grid_lat, mslp_grid, "max",
                       n_size=N_SIZE, symbol_size=SYMBOL_SIZE, transform=data_crs)
    plot_maxmin_points(ax, grid_lon, grid_lat, mslp_grid, "min",
                       n_size=N_SIZE, symbol_size=SYMBOL_SIZE, transform=data_crs)

    title_line = f"MSLP Isobars with Pressure Centers (H/L)\n{time_label}"
    ax.set_title(title_line, fontsize=FONT_TITLE, loc="left", fontweight="bold")
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"europe_isobars_hl_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.show()

    return str(out_path.resolve())


def plot_europe_enhanced_station_isobars(
    df_plot: pd.DataFrame,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    mslp_grid: np.ndarray,
    output_dir: str = "../outputs",
    isobar_interval: int = 4,
) -> str:
    """
    Draw an enhanced European station plot with temperature labels, wind
    barbs, isobars, and H/L pressure centers.

    The observation time is derived automatically from the ``valid`` column
    of *df_plot* and included in the map title.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Plot-ready observation DataFrame from ``build_europe_network_plot_df()``.
        Must contain ``lon``, ``lat``, ``valid``, ``temp_c``,
        ``wdir``, ``wspd``, ``u_kt``, ``v_kt``.
    grid_lon : np.ndarray
        1-D array of grid longitudes (degrees East).
    grid_lat : np.ndarray
        1-D array of grid latitudes (degrees North).
    mslp_grid : np.ndarray
        2-D array of smoothed MSLP values (hPa), shape
        ``(len(grid_lat), len(grid_lon))``.
    output_dir : str
        Directory to save the figure.
    isobar_interval : int
        Contour interval in hPa. Default 4 hPa (WMO standard).

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    # ── Derive time label from observations ───────────────────────────────
    obs_times = pd.to_datetime(df_plot["valid"]).dropna()
    if len(obs_times) > 0:
        t_min, t_max = obs_times.min(), obs_times.max()
        if t_min.strftime("%Y-%m-%d %H") == t_max.strftime("%Y-%m-%d %H"):
            time_label = f"Latest METAR: {t_max.strftime('%Y-%m-%d %H:%M UTC')}"
        else:
            time_label = (f"METAR range: {t_min.strftime('%Y-%m-%d %H:%M')} "
                          f"\u2013 {t_max.strftime('%H:%M')} UTC")
    else:
        time_label = "Latest METAR: unknown time"

    fig, ax, proj, data_crs = _setup_europe_map()

    if T_THINNING_KM > 0:
        proj_points = proj.transform_points(data_crs, df_plot["lon"].values, df_plot["lat"].values)
        mask = reduce_point_density(proj_points, T_THINNING_KM * 1000)
        df_thin = df_plot[mask].copy()
    else:
        df_thin = df_plot.copy()

    # Draw isobars (bolder)
    isobar_levels = np.arange(900, 1060, isobar_interval)
    cs = ax.contour(
        grid_lon, grid_lat, mslp_grid,
        levels=isobar_levels, colors="black", linewidths=1.2,
        alpha=0.8, transform=data_crs,
    )
    ax.clabel(cs, cs.levels[::2], fmt="%d", fontsize=max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR)), inline=True)

    # Wind barbs & points
    is_vrb  = df_thin["wdir"].isna()
    is_calm = (~is_vrb) & (df_thin["wspd"].fillna(0) <= CALM_THRESHOLD_KT)
    is_barb = (~is_vrb) & (~is_calm)

    if is_barb.any():
        ax.barbs(
            df_thin.loc[is_barb, "lon"],
            df_thin.loc[is_barb, "lat"],
            df_thin.loc[is_barb, "u_kt"],
            df_thin.loc[is_barb, "v_kt"],
            length=BARB_LENGTH * BARB_SCALE_STATION, transform=data_crs, color="black", linewidth=0.6,
        )

    if is_vrb.any():
        ax.scatter(
            df_thin.loc[is_vrb, "lon"],
            df_thin.loc[is_vrb, "lat"],
            s=np.clip(df_thin.loc[is_vrb, "wspd"] * VRB_SCATTER_MULTIPLIER_ENHANCED, VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
            c="black", transform=data_crs, label="VRB",
        )

    if is_calm.any():
        ax.scatter(
            df_thin.loc[is_calm, "lon"],
            df_thin.loc[is_calm, "lat"],
            s=CALM_CIRCLE_S, facecolors="none", edgecolors="black",
            transform=data_crs, label="Calm",
        )

    # Temperature labels
    for _, rr in df_thin.iterrows():
        if pd.notna(rr.get("temp_c", float("nan"))):
            ax.text(
                rr["lon"] + LABEL_LON_OFFSET,
                rr["lat"] + LABEL_LAT_UPPER,
                f"{rr['temp_c']:.0f} \u00b0C",
                color="red", fontsize=FONT_LABEL, fontweight="bold",
                transform=data_crs, clip_on=True,
            )

    # H/L centers
    plot_maxmin_points(ax, grid_lon, grid_lat, mslp_grid, "max",
                       n_size=N_SIZE, symbol_size=SYMBOL_SIZE, transform=data_crs)
    plot_maxmin_points(ax, grid_lon, grid_lat, mslp_grid, "min",
                       n_size=N_SIZE, symbol_size=SYMBOL_SIZE, transform=data_crs)

    title_line = f"Surface Temperature, Wind, and Pressure Centers\n{time_label}"
    ax.set_title(title_line, fontsize=FONT_TITLE, loc="left", fontweight="bold")
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"week2_frontal_overview_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.show()

    return str(out_path.resolve())


# [Progressive frontal-zone maps — Sections 2.2.4, 3.2.1, 3.2.2]

def plot_europe_isobars_wind(
    df_plot: pd.DataFrame,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    mslp_grid: np.ndarray,
    output_dir: str = "../outputs",
    isobar_interval: int = 4,
) -> str:
    """
    Draw a map with MSLP isobars, H/L pressure centers, and observed wind barbs.

    The observation time is derived automatically from the ``valid`` column
    of *df_plot* and included in the map title.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Plot-ready observation DataFrame from ``build_europe_network_plot_df()``.
        Must contain ``lon``, ``lat``, ``valid``, ``wdir``,
        ``wspd``, ``u_kt``, ``v_kt``.
    grid_lon : np.ndarray
        1-D array of grid longitudes (degrees East).
    grid_lat : np.ndarray
        1-D array of grid latitudes (degrees North).
    mslp_grid : np.ndarray
        2-D array of smoothed MSLP values (hPa), shape
        ``(len(grid_lat), len(grid_lon))``.
    output_dir : str
        Directory to save the figure.
    isobar_interval : int
        Contour interval in hPa. Default 4 hPa (WMO standard).

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    # ── Derive time label ─────────────────────────────────────────────────
    obs_times = pd.to_datetime(df_plot["valid"]).dropna()
    if len(obs_times) > 0:
        t_min, t_max = obs_times.min(), obs_times.max()
        if t_min.strftime("%Y-%m-%d %H") == t_max.strftime("%Y-%m-%d %H"):
            time_label = f"Latest METAR: {t_max.strftime('%Y-%m-%d %H:%M UTC')}"
        else:
            time_label = (f"METAR range: {t_min.strftime('%Y-%m-%d %H:%M')} "
                          f"\u2013 {t_max.strftime('%H:%M')} UTC")
    else:
        time_label = "Latest METAR: unknown time"

    fig, ax, proj, data_crs = _setup_europe_map()

    if THINNING_RADIUS_KM > 0:
        proj_points = proj.transform_points(data_crs, df_plot["lon"].values, df_plot["lat"].values)
        mask = reduce_point_density(proj_points, THINNING_RADIUS_KM * 1000)
        df_thin = df_plot[mask].copy()
    else:
        df_thin = df_plot.copy()

    # ── Isobars and H/L centers ───────────────────────────────────────────
    isobar_levels = np.arange(900, 1060, isobar_interval)
    cs = ax.contour(
        grid_lon, grid_lat, mslp_grid,
        levels=isobar_levels, colors="black", linewidths=1.2,
        transform=data_crs,
    )
    ax.clabel(cs, cs.levels[::2], fmt="%d", fontsize=max(1, round(FONT_LABEL)), inline=True)

    plot_maxmin_points(ax, grid_lon, grid_lat, mslp_grid, "max",
                       n_size=N_SIZE, symbol_size=SYMBOL_SIZE, transform=data_crs)
    plot_maxmin_points(ax, grid_lon, grid_lat, mslp_grid, "min",
                       n_size=N_SIZE, symbol_size=SYMBOL_SIZE, transform=data_crs)

    # ── Wind barbs & points ───────────────────────────────────────────────
    is_vrb  = df_thin["wdir"].isna()
    is_calm = (~is_vrb) & (df_thin["wspd"].fillna(0) <= CALM_THRESHOLD_KT)
    is_barb = (~is_vrb) & (~is_calm)

    if is_barb.any():
        ax.barbs(
            df_thin.loc[is_barb, "lon"],
            df_thin.loc[is_barb, "lat"],
            df_thin.loc[is_barb, "u_kt"],
            df_thin.loc[is_barb, "v_kt"],
            length=BARB_LENGTH * BARB_SCALE_STATION, transform=data_crs, color="black",
        )
    if is_vrb.any():
        ax.scatter(
            df_thin.loc[is_vrb, "lon"],
            df_thin.loc[is_vrb, "lat"],
            s=np.clip(df_thin.loc[is_vrb, "wspd"] * VRB_SCATTER_MULTIPLIER, VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
            c="black", transform=data_crs,
        )
    if is_calm.any():
        ax.scatter(
            df_thin.loc[is_calm, "lon"],
            df_thin.loc[is_calm, "lat"],
            s=CALM_CIRCLE_S, facecolors="none", edgecolors="black",
            transform=data_crs,
        )

    ax.set_title(
        f"MSLP Isobars, Pressure Centers, and Wind\n{time_label}",
        fontsize=FONT_TITLE, loc="left", fontweight="bold",
    )
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"europe_isobars_wind_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.show()

    return str(out_path.resolve())


def plot_europe_isobars_temperature(
    df_plot: pd.DataFrame,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    mslp_grid: np.ndarray,
    temp_grid: np.ndarray,
    output_dir: str = "../outputs",
    isobar_interval: int = 4,
    isotherm_interval: int = 2,
) -> str:
    """
    Two-panel figure: (left) MSLP isobars, H/L centers, and thinned wind barbs;
    (right) isobars, H/L centers, thinned wind barbs, and temperature isotherms.

    The observation time is derived automatically from the ``valid`` column
    of *df_plot* and included in each panel title.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Plot-ready observation DataFrame from ``build_europe_network_plot_df()``.
        Must contain ``lon``, ``lat``, ``valid``, ``wdir``,
        ``wspd``, ``u_kt``, ``v_kt``.
    grid_lon : np.ndarray
        1-D array of grid longitudes (degrees East).
    grid_lat : np.ndarray
        1-D array of grid latitudes (degrees North).
    mslp_grid : np.ndarray
        2-D array of smoothed MSLP values (hPa), shape
        ``(len(grid_lat), len(grid_lon))``.
    temp_grid : np.ndarray
        2-D array of smoothed temperature values (°C), same shape as
        *mslp_grid*.
    output_dir : str
        Directory to save the figure.
    isobar_interval : int
        MSLP contour interval in hPa. Default 4 hPa (WMO standard).
    isotherm_interval : int
        Temperature contour interval in °C. Default 2 °C.

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    # ── Derive time label ─────────────────────────────────────────────────
    obs_times = pd.to_datetime(df_plot["valid"]).dropna()
    if len(obs_times) > 0:
        t_min, t_max = obs_times.min(), obs_times.max()
        if t_min.strftime("%Y-%m-%d %H") == t_max.strftime("%Y-%m-%d %H"):
            time_label = f"Latest METAR: {t_max.strftime('%Y-%m-%d %H:%M UTC')}"
        else:
            time_label = (f"METAR range: {t_min.strftime('%Y-%m-%d %H:%M')} "
                          f"\u2013 {t_max.strftime('%H:%M')} UTC")
    else:
        time_label = "Latest METAR: unknown time"

    # ── Panel-scaled font/symbol sizes (~75 % of single-panel constants) ─
    _ft = max(1, round(FONT_TITLE))
    _fl = max(1, round(FONT_LABEL))
    _fs = max(1, round(SYMBOL_SIZE))
    _bl = BARB_LENGTH * BARB_SCALE_STATION 

    # ── Two-panel figure (double width, same height as single-panel) ─────
    proj = ccrs.LambertConformal(
        central_longitude=PROJ_CENTRAL_LON,
        central_latitude=PROJ_CENTRAL_LAT,
        standard_parallels=PROJ_STD_PARALLELS,
    )
    data_crs = ccrs.PlateCarree()

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(FIG_SIZE_IN[0] * 2, FIG_SIZE_IN[1]),
        subplot_kw={"projection": proj},
        gridspec_kw={"wspace": SUBPLOT_WSPACE_OVERVIEW},
    )
    fig.patch.set_facecolor("w")

    _setup_europe_map(ax=ax1)
    _setup_europe_map(ax=ax2)

    # ── Thinning (shared between both panels) ────────────────────────────
    if THINNING_RADIUS_KM > 0:
        proj_points = proj.transform_points(
            data_crs, df_plot["lon"].values, df_plot["lat"].values
        )
        mask = reduce_point_density(proj_points, THINNING_RADIUS_KM * 1000)
        df_thin = df_plot[mask].copy()
    else:
        df_thin = df_plot.copy()

    # ── Panel 1 — MSLP isobars + H/L centers + wind barbs ────────────────
    isobar_levels = np.arange(900, 1060, isobar_interval)
    cs = ax1.contour(
        grid_lon, grid_lat, mslp_grid,
        levels=isobar_levels, colors="black", linewidths=1.2,
        transform=data_crs,
    )
    ax1.clabel(cs, cs.levels[::2], fmt="%d", fontsize=_fl, inline=True)

    plot_maxmin_points(ax1, grid_lon, grid_lat, mslp_grid, "max",
                       n_size=N_SIZE, symbol_size=_fs, transform=data_crs)
    plot_maxmin_points(ax1, grid_lon, grid_lat, mslp_grid, "min",
                       n_size=N_SIZE, symbol_size=_fs, transform=data_crs)

    is_vrb  = df_thin["wdir"].isna()
    is_calm = (~is_vrb) & (df_thin["wspd"].fillna(0) <= CALM_THRESHOLD_KT)
    is_barb = (~is_vrb) & (~is_calm)

    if is_barb.any():
        ax1.barbs(
            df_thin.loc[is_barb, "lon"],
            df_thin.loc[is_barb, "lat"],
            df_thin.loc[is_barb, "u_kt"],
            df_thin.loc[is_barb, "v_kt"],
            length=_bl, transform=data_crs, color="black",
        )
    if is_vrb.any():
        ax1.scatter(
            df_thin.loc[is_vrb, "lon"],
            df_thin.loc[is_vrb, "lat"],
            s=np.clip(df_thin.loc[is_vrb, "wspd"] * VRB_SCATTER_MULTIPLIER, VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
            c="black", transform=data_crs,
        )
    if is_calm.any():
        ax1.scatter(
            df_thin.loc[is_calm, "lon"],
            df_thin.loc[is_calm, "lat"],
            s=CALM_CIRCLE_S, facecolors="none", edgecolors="black",
            transform=data_crs,
        )

    ax1.set_title(
        f"MSLP Isobars, Pressure Centers, and Wind\n{time_label}",
        fontsize=_ft, loc="left", fontweight="bold",
    )

    # ── Panel 2 — temperature isotherms only ─────────────────────────────
    isotherm_levels = np.arange(TEMP_ISOTHERM_MIN, TEMP_ISOTHERM_MAX, isotherm_interval)    
    cs_t = ax2.contour(
        grid_lon, grid_lat, temp_grid,
        levels=isotherm_levels, colors="#cc0000", linewidths=1.4,
        alpha=1.0, transform=data_crs,
    )
    ax2.clabel(cs_t, cs_t.levels[::2], fmt="%.0f", fontsize=_fl, inline=True,
               colors="#cc0000")

    ax2.set_title(
        f"Temperature Isotherms ({isotherm_interval} °C interval)\n{time_label}",
        fontsize=_ft, loc="left", fontweight="bold",
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"europe_isobars_temperature_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.show()

    return str(out_path.resolve())


def plot_europe_isobars_temperature_humidity(
    df_plot: pd.DataFrame,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    mslp_grid: np.ndarray,
    temp_grid: np.ndarray,
    output_dir: str = "../outputs",
    isobar_interval: int = 4,
    isotherm_interval: int = 2,
) -> str:
    """
    Two-panel figure extending ``plot_europe_isobars_temperature``:

    * Left: MSLP isobars, H/L pressure centers, thinned wind barbs, and
      thinned station relative humidity (%) in blue at the lower-left of each
      station position, matching the layout of ``plot_europe_metar_network``.
    * Right: temperature isotherms only (red).

    The observation time is derived automatically from the ``valid`` column
    of *df_plot* and included in each panel title.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Plot-ready observation DataFrame from ``build_europe_network_plot_df()``.
        Must contain ``lon``, ``lat``, ``valid``, ``wdir``,
        ``wspd``, ``u_kt``, ``v_kt``, ``relh``.
    grid_lon : np.ndarray
        1-D array of grid longitudes (degrees East).
    grid_lat : np.ndarray
        1-D array of grid latitudes (degrees North).
    mslp_grid : np.ndarray
        2-D array of smoothed MSLP values (hPa), shape
        ``(len(grid_lat), len(grid_lon))``.
    temp_grid : np.ndarray
        2-D array of smoothed temperature values (°C), same shape as
        *mslp_grid*.
    output_dir : str
        Directory to save the figure.
    isobar_interval : int
        MSLP contour interval in hPa. Default 4 hPa (WMO standard).
    isotherm_interval : int
        Temperature contour interval in °C. Default 2 °C.

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    # ── Derive time label ─────────────────────────────────────────────────
    obs_times = pd.to_datetime(df_plot["valid"]).dropna()
    if len(obs_times) > 0:
        t_min, t_max = obs_times.min(), obs_times.max()
        if t_min.strftime("%Y-%m-%d %H") == t_max.strftime("%Y-%m-%d %H"):
            time_label = f"Latest METAR: {t_max.strftime('%Y-%m-%d %H:%M UTC')}"
        else:
            time_label = (f"METAR range: {t_min.strftime('%Y-%m-%d %H:%M')} "
                          f"\u2013 {t_max.strftime('%H:%M')} UTC")
    else:
        time_label = "Latest METAR: unknown time"

    # ── Panel font/symbol sizes ───────────────────────────────────────────
    _ft  = max(1, round(FONT_TITLE))
    _fl  = max(1, round(FONT_LABEL ))
    _fs  = max(1, round(SYMBOL_SIZE))
    _bl  = BARB_LENGTH * BARB_SCALE_STATION
    _rfs = max(1, round(FONT_LABEL  * FONT_SCALE_STATION))

    proj = ccrs.LambertConformal(
        central_longitude=PROJ_CENTRAL_LON,
        central_latitude=PROJ_CENTRAL_LAT,
        standard_parallels=PROJ_STD_PARALLELS,
    )
    data_crs = ccrs.PlateCarree()

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(FIG_SIZE_IN[0] * 2, FIG_SIZE_IN[1]),
        subplot_kw={"projection": proj},
        gridspec_kw={"wspace": SUBPLOT_WSPACE_OVERVIEW},
    )
    fig.patch.set_facecolor("w")

    _setup_europe_map(ax=ax1)
    _setup_europe_map(ax=ax2)

    # ── Thinning (shared between both panels) ────────────────────────────
    if THINNING_RADIUS_KM > 0:
        proj_points = proj.transform_points(
            data_crs, df_plot["lon"].values, df_plot["lat"].values
        )
        mask = reduce_point_density(proj_points, THINNING_RADIUS_KM * 1000)
        df_thin = df_plot[mask].copy()
    else:
        df_thin = df_plot.copy()

    is_vrb  = df_thin["wdir"].isna()
    is_calm = (~is_vrb) & (df_thin["wspd"].fillna(0) <= CALM_THRESHOLD_KT)
    is_barb = (~is_vrb) & (~is_calm)

    # ── Panel 1 — MSLP isobars + H/L + thinned wind barbs + RH labels ───
    isobar_levels = np.arange(900, 1060, isobar_interval)
    cs = ax1.contour(
        grid_lon, grid_lat, mslp_grid,
        levels=isobar_levels, colors="black", linewidths=1.2,
        transform=data_crs,
    )
    ax1.clabel(cs, cs.levels[::2], fmt="%d", fontsize=_fl, inline=True)

    plot_maxmin_points(ax1, grid_lon, grid_lat, mslp_grid, "max",
                       n_size=N_SIZE, symbol_size=_fs, transform=data_crs)
    plot_maxmin_points(ax1, grid_lon, grid_lat, mslp_grid, "min",
                       n_size=N_SIZE, symbol_size=_fs, transform=data_crs)

    if is_barb.any():
        ax1.barbs(
            df_thin.loc[is_barb, "lon"], df_thin.loc[is_barb, "lat"],
            df_thin.loc[is_barb, "u_kt"], df_thin.loc[is_barb, "v_kt"],
            length=_bl, transform=data_crs, color="black",
        )
    if is_vrb.any():
        ax1.scatter(
            df_thin.loc[is_vrb, "lon"], df_thin.loc[is_vrb, "lat"],
            s=np.clip(df_thin.loc[is_vrb, "wspd"] * VRB_SCATTER_MULTIPLIER, VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
            c="black", transform=data_crs,
        )
    if is_calm.any():
        ax1.scatter(
            df_thin.loc[is_calm, "lon"], df_thin.loc[is_calm, "lat"],
            s=CALM_CIRCLE_S, facecolors="none", edgecolors="black", transform=data_crs,
        )

    for _, rr in df_thin.iterrows():
        if pd.notna(rr.get("relh", np.nan)):
            ax1.text(
                rr["lon"] - LABEL_LON_OFFSET * LABEL_SCALE_STATION,
                rr["lat"]  - LABEL_LAT_LOWER  * LABEL_SCALE_STATION,
                f"{rr['relh']:.0f}%",
                color="blue", fontsize=_rfs, ha="right",
                transform=data_crs, clip_on=True,
            )

    ax1.set_title(
        f"MSLP Isobars, Pressure Centers, Wind, and Relative Humidity\n{time_label}",
        fontsize=_ft, loc="left", fontweight="bold",
    )

    # ── Panel 2 — temperature isotherms only ─────────────────────────────
    isotherm_levels = np.arange(TEMP_ISOTHERM_MIN, TEMP_ISOTHERM_MAX, isotherm_interval)
    cs_t = ax2.contour(
        grid_lon, grid_lat, temp_grid,
        levels=isotherm_levels, colors="#cc0000", linewidths=1.4,
        transform=data_crs,
    )
    ax2.clabel(cs_t, cs_t.levels[::2], fmt="%.0f", fontsize=_fl, inline=True,
               colors="#cc0000")

    ax2.set_title(
        f"Temperature Isotherms ({isotherm_interval} °C interval)\n{time_label}",
        fontsize=_ft, loc="left", fontweight="bold",
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"europe_isobars_temperature_humidity_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.show()

    return str(out_path.resolve())

def plot_gfs_surface_chart(
    ds: xr.Dataset,
    mslp_interval: float = MSLP_INTERVAL,
    sigma_mslp: float = SIGMA_SURFACE,
    t2m_interval: float = T2M_INTERVAL,
    output_dir: str = "../outputs",
) -> str:
    """
    GFS-analysis surface chart: MSLP isobars, H/L centers, 10 m wind barbs,
    and 2 m temperature contours (red dashed).

    Parameters
    ----------
    ds : xr.Dataset
        GFS analysis dataset (must contain ``t_2m`` in Kelvin).
    mslp_interval : float
        MSLP contour interval (hPa).
    sigma_mslp : float
        Gaussian smoothing sigma applied to both MSLP and T2m.
    t2m_interval : float
        2 m temperature contour interval (°C).
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    valid_time = ds.attrs.get("valid_time", "unknown")
    lon        = ds["longitude"].values
    lat        = ds["latitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    mslp_hpa = _smooth_field(ds["prmsl"].values / 100.0, sigma_mslp)

    fig, ax, proj, data_crs = _setup_europe_map(proj=ccrs.PlateCarree())

    # ── MSLP isobars ────────────────────────────────────────────────────────
    p_min  = np.floor(np.nanmin(mslp_hpa) / mslp_interval) * mslp_interval
    p_max  = np.ceil( np.nanmax(mslp_hpa) / mslp_interval) * mslp_interval
    p_levs = np.arange(p_min, p_max + mslp_interval, mslp_interval)
    cs_p   = ax.contour(
        lon_2d, lat_2d, mslp_hpa,
        levels=p_levs, colors="black", linewidths=1.2,
        transform=data_crs,
    )
    ax.clabel(
        cs_p, cs_p.levels[::GPH_LABEL_STRIDE], fmt="%d",
        fontsize=max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR)), inline=True,
    )

    # ── H/L pressure centers ────────────────────────────────────────────────
    plot_maxmin_points(ax, lon, lat, mslp_hpa, "max",
                       n_size=N_SIZE, symbol_size=SYMBOL_SIZE,
                       min_sep_deg=HL_MIN_SEP_DEG, transform=data_crs)
    plot_maxmin_points(ax, lon, lat, mslp_hpa, "min",
                       n_size=N_SIZE, symbol_size=SYMBOL_SIZE,
                       min_sep_deg=HL_MIN_SEP_DEG, transform=data_crs)

    # ── 2 m temperature contours (red dashed) ───────────────────────────────
    t2m_c  = _smooth_field(ds["t_2m"].values - 273.15, sigma_mslp)
    t_min  = np.floor(np.nanmin(t2m_c) / t2m_interval) * t2m_interval
    t_max  = np.ceil( np.nanmax(t2m_c) / t2m_interval) * t2m_interval
    t_levs = np.arange(t_min, t_max + t2m_interval, t2m_interval)
    cs_t   = ax.contour(
        lon_2d, lat_2d, t2m_c,
        levels=t_levs, colors="red", linewidths=0.8,
        linestyles="dashed", transform=data_crs,
    )
    ax.clabel(
        cs_t, cs_t.levels[::2], fmt="%d°C",
        fontsize=max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR)), inline=True,
    )

    # ── 10 m wind barbs ─────────────────────────────────────────────────────
    st = _BARB_STRIDE
    ax.barbs(
        lon_2d[::st, ::st], lat_2d[::st, ::st],
        ds["u_10m"].values[::st, ::st], ds["v_10m"].values[::st, ::st],
        length=BARB_LENGTH * BARB_SCALE_UPPER,
        transform=data_crs, color="black",
    )

    ax.set_title(
        _gfs_title("MSLP [black contours, hPa], 10 m Wind [black barbs], "
                   "and 2 m Temperature [red dashed contours, °C]", valid_time),
        fontsize=FONT_TITLE * FONT_SCALE_UPPER_TITLE, loc="left", fontweight="bold",
    )

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.to_datetime("now", utc=True).strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"gfs_surface_chart_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    return str(out_path.resolve())


# [GEFS ensemble surface chart]

def plot_ensemble_mean_spread(
    mean_field: xr.DataArray,
    spread_field: xr.DataArray,
    domain: tuple[float, float, float, float],
    lead_time_h: int,
    init_date: str,
    output_dir: str | Path = "../outputs",
    dpi: int = FIG_DPI_OVERVIEW,
) -> Path:
    """
    Two-panel static map of GEFS ensemble mean and spread for MSLP.

    Left panel: ensemble mean MSLP contoured at MSLP_INTERVAL interval with
    labeled isobars and H/L pressure centers. Right panel: ensemble standard
    deviation shaded with a perceptually uniform colormap; faint mean isobars
    for spatial reference.

    Parameters
    ----------
    mean_field : xr.DataArray
        Ensemble mean MSLP at a single lead time (Pa), dims (latitude, longitude).
    spread_field : xr.DataArray
        Ensemble standard deviation of MSLP (Pa), same dims.
    domain : tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees. The map extent
        is controlled by ``_setup_europe_map`` (``EUROPE_EXTENT``); this
        parameter is retained for API compatibility.
    lead_time_h : int
        Lead time in hours; used in the panel titles.
    init_date : str
        Initialization date string; used in the panel titles.
    output_dir : str or Path
        Directory for the saved PNG. Created if absent.
    dpi : int
        Output resolution (dots per inch).

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_crs = ccrs.PlateCarree()

    # Pa → hPa, NaN-safe smoothing
    mean_np   = _smooth_field(mean_field.values   / 100.0, SIGMA_SURFACE)
    spread_np = _smooth_field(spread_field.values / 100.0, SIGMA_SURFACE)

    lat = mean_field.latitude.values
    lon = mean_field.longitude.values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    p_min     = np.floor(np.nanmin(mean_np) / MSLP_INTERVAL) * MSLP_INTERVAL
    p_max     = np.ceil( np.nanmax(mean_np) / MSLP_INTERVAL) * MSLP_INTERVAL
    mean_levs = np.arange(p_min, p_max + MSLP_INTERVAL, MSLP_INTERVAL)

    proj = ccrs.PlateCarree()
    fig, (ax_mean, ax_spread) = plt.subplots(
        1, 2,
        figsize=(FIG_SIZE_IN[0] * 2, FIG_SIZE_IN[1]),
        subplot_kw={"projection": proj},
        gridspec_kw={"wspace": SUBPLOT_WSPACE_OVERVIEW_FORECAST},
    )
    fig.patch.set_facecolor("w")

    _setup_europe_map(ax=ax_mean)
    _setup_europe_map(ax=ax_spread)

    # ── Left panel: ensemble mean MSLP isobars + H/L centers ────────────────
    cs_mean = ax_mean.contour(
        lon_2d, lat_2d, mean_np,
        levels=mean_levs, colors="black", linewidths=1.2,
        transform=data_crs,
    )
    ax_mean.clabel(
        cs_mean, cs_mean.levels[::GPH_LABEL_STRIDE], fmt="%d",
        fontsize=max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR)), inline=True,
    )
    plot_maxmin_points(ax_mean, lon, lat, mean_np, "max",
                       n_size=N_SIZE, symbol_size=SYMBOL_SIZE,
                       min_sep_deg=HL_MIN_SEP_DEG, transform=data_crs)
    plot_maxmin_points(ax_mean, lon, lat, mean_np, "min",
                       n_size=N_SIZE, symbol_size=SYMBOL_SIZE,
                       min_sep_deg=HL_MIN_SEP_DEG, transform=data_crs)
    ax_mean.set_title(
        _gefs_title("GEFS Ensemble Mean MSLP [contours, hPa]", init_date, lead_time_h),
        fontsize=FONT_TITLE * FONT_SCALE_UPPER_TITLE, loc="left", fontweight="bold",
    )
    _div_mean = make_axes_locatable(ax_mean)
    _cax_mean = _div_mean.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD,
                                      axes_class=plt.Axes)
    _cax_mean.set_visible(False)

    # ── Right panel: ensemble spread shaded + faint mean contours ───────────
    spread_max  = max(float(np.nanmax(spread_np)), 1.0)
    # Shading starts at 1.0 hPa; values below are unshaded (background).
    spread_levs = np.arange(1.0, spread_max + 0.5, 0.5)
    im_spread = ax_spread.contourf(
        lon_2d, lat_2d, spread_np,
        levels=spread_levs, cmap="Purples", transform=data_crs,
        extend="max",
    )
    ax_spread.contour(
        lon_2d, lat_2d, mean_np,
        levels=mean_levs[::2], colors="black", linewidths=0.4,
        alpha=0.4, transform=data_crs,
    )
    ax_spread.set_title(
        _gefs_title("GEFS Ensemble Spread MSLP [std dev, hPa]", init_date, lead_time_h),
        fontsize=FONT_TITLE * FONT_SCALE_UPPER_TITLE, loc="left", fontweight="bold",
    )
    _div_spread = make_axes_locatable(ax_spread)
    _cax_spread = _div_spread.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD,
                                          axes_class=plt.Axes)
    cb = fig.colorbar(im_spread, cax=_cax_spread)
    cbar_ticks = np.arange(1.0, spread_max + 0.5, 0.5)
    cb.set_ticks(cbar_ticks[cbar_ticks <= spread_max])
    cb.set_label("hPa", fontsize=max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR)))
    cb.ax.tick_params(labelsize=max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR)))

    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"gefs_mslp_mean_spread_{timestamp}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")

    return out_path.resolve()


def plot_compound_probability_animation(
    compound_prob: xr.DataArray,
    domain: tuple[float, float, float, float],
    init_date: str,
    max_signal_lead_h: int,
    output_dir: str | Path,
    case: str = "",
    dpi: int = FIG_DPI_OVERVIEW,
) -> Path:
    """
    Animate the compound probability field across the forecast horizon.

    Parameters
    ----------
    compound_prob : xr.DataArray
        Compound probability field, dims (valid_time, latitude, longitude).
    domain : tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees. Retained for
        API compatibility; map extent is controlled by ``_setup_europe_map``.
    init_date : str
        Initialization date string; used in frame titles.
    max_signal_lead_h : int
        Lead time of maximum compound signal; annotated on the relevant frame.
    output_dir : str or Path
        Directory for the exported GIF and frame PNGs. Created if absent.
    case : str
        Active case label from ``CASE_DEFINITIONS`` (e.g. ``"severe_convection"``);
        appended to the title as ``"Compound Probability Product: {case}"``.
    dpi : int
        Output resolution (dots per inch). Defaults to ``FIG_DPI_OVERVIEW``.

    Returns
    -------
    Path
        Absolute path to the saved animated GIF. Individual frame PNGs are
        also saved to ``output_dir`` as
        ``compound_probability_{init_yyyymmdd_hhmm}_f{NNN}h.png``.
    """
    _fs = max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR * FONT_SCALE_TITLE_OVERVIEW))
    _fs_prob_cbar = _fs + 2  

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_crs  = ccrs.PlateCarree()
    lat       = compound_prob.latitude.values
    lon       = compound_prob.longitude.values
    lead_vals = compound_prob.valid_time.values

    compound_max = max(float(compound_prob.max()), 1e-6)
    cp_levels    = np.linspace(0, compound_max, 21)[1:]

    fig = plt.figure(figsize=FIG_SIZE_IN)
    fig.patch.set_facecolor("w")
    ax = fig.add_subplot(1, 1, 1, projection=data_crs)

    _setup_europe_map(ax=ax)
    first_lead = int(lead_vals[0])
    im = ax.contourf(
        lon, lat, compound_prob.sel(valid_time=first_lead).values,
        levels=cp_levels, cmap="YlOrRd", transform=data_crs, extend="max",
    )
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD, axes_class=plt.Axes)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label("Compound probability [%]", fontsize=_fs)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}"))
    cb.ax.tick_params(labelsize=_fs_prob_cbar)

    init_safe = pd.Timestamp(init_date).strftime("%Y%m%d_%H%M")
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"compound_probability_animation_{timestamp}.gif"
    gif_frames: list[Image.Image] = []

    for lead_h in [int(lt) for lt in lead_vals]:
        ax.cla()
        _setup_europe_map(ax=ax)

        ax.contourf(
            lon, lat, compound_prob.sel(valid_time=lead_h).values,
            levels=cp_levels, cmap="YlOrRd", transform=data_crs, extend="max",
        )

        _field_label = (
            f"Compound Probability Product: {case}" if case
            else "Compound Probability Product"
        )
        title = _gefs_title(_field_label, init_date, lead_h)
        ax.set_title(
            title,
            fontsize=FONT_TITLE * FONT_SCALE_UPPER_TITLE,
            loc="left",
            fontweight="bold",
        )

        frame_path = out_dir / f"compound_probability_{init_safe}_f{lead_h:03d}h.png"
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        buf.seek(0)
        gif_frames.append(Image.open(buf).copy())

    gif_frames[0].save(
        str(out_path),
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        duration=_GIF_INTERVAL_MS,
        loop=0,
        optimize=False,
    )
    plt.close(fig)

    return out_path.resolve()


# [Upper-air station plot]

def plot_europe_500hpa_stations(
    df_raob,
    output_dir: str = "../outputs",
) -> str:
    """
    Upper-air station plot at 500 hPa for the European radiosonde network.

    Each station is drawn with:
    - Upper-left: temperature (°C, red).
    - Upper-right: geopotential height in decameters (dam), coded as the
                    last 3 digits (e.g. 576 for 5760 m).
    - Lower-left: dew-point depression (T - Td, °C), rounded to 1 °C.
    - Wind barbs: 500 hPa wind in knots.

    Parameters
    ----------
    df_raob : pd.DataFrame
        Output of fetch_europe_raob_fields().  Must contain
        longitude, latitude, z500_dam, t500, dd500,
        u500, v500, valid.
    output_dir : str
        Directory to save the PNG.

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    proj     = ccrs.LambertConformal(
        central_longitude=PROJ_CENTRAL_LON,
        central_latitude=PROJ_CENTRAL_LAT,
        standard_parallels=PROJ_STD_PARALLELS,
    )
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=FIG_SIZE_IN)
    fig.patch.set_facecolor("w")
    ax = plt.axes(projection=proj)
    ax.set_extent(EUROPE_EXTENT, crs=data_crs)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),   linewidth=0.3)
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="lightgray", alpha=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray",
                 alpha=0.4, linestyle="--")

    _fs = max(1, round(FONT_LABEL * FONT_SCALE_STATION))
    _lo = LABEL_LON_OFFSET * LABEL_SCALE_STATION
    _lu = RAOB_LABEL_LAT_UPPER
    _ll = RAOB_LABEL_LAT_LOWER

    # Thin stations to avoid crowding in dense regions (e.g. Central Europe)
    if RAOB_THINNING_RADIUS_KM > 0:
        proj_points = proj.transform_points(
            data_crs,
            df_raob["longitude"].values,
            df_raob["latitude"].values,
        )
        mask = reduce_point_density(proj_points, RAOB_THINNING_RADIUS_KM * 1000)
        df_plot = df_raob[mask].copy()
    else:
        df_plot = df_raob.copy()

    df_wind = df_plot.dropna(subset=["u500", "v500"])
    if not df_wind.empty:
        ax.barbs(
            df_wind["longitude"].values,
            df_wind["latitude"].values,
            df_wind["u500"].values,
            df_wind["v500"].values,
            length=BARB_LENGTH * BARB_SCALE_500_RAOB,
            transform=data_crs, color="black",
        )

    for _, row in df_plot.iterrows():
        lon, lat = row["longitude"], row["latitude"]

        # Temperature label — upper left (red)
        if pd.notna(row.get("t500", np.nan)):
            ax.text(
                lon - _lo, lat + _lu,
                f"{row['t500']:.0f}",
                color="red", fontsize=_fs,
                ha="right", transform=data_crs, clip_on=True,
            )

        # GPH label — upper right, last 3 digits of dam value
        if pd.notna(row.get("z500_dam", np.nan)):
            coded = int(round(row["z500_dam"])) % 1000
            ax.text(
                lon + _lo, lat + _lu,
                f"{coded:03d}",
                color="black", fontsize=_fs,
                ha="left", transform=data_crs, clip_on=True,
            )

        # Dew-point depression label — lower left
        if pd.notna(row.get("dd500", np.nan)):
            ax.text(
                lon - _lo, lat - _ll,
                f"{row['dd500']:.0f}",
                color="blue", fontsize=_fs,
                ha="right", transform=data_crs, clip_on=True,
            )

    # Time label
    valid_times = pd.to_datetime(df_raob["valid"]).dropna()
    time_label = (valid_times.max().strftime("%Y-%m-%d %H:%M UTC")
              if len(valid_times) > 0 else "unknown time")

    ax.set_title(
        f"500 hPa Station Plot\nValid time: {time_label}",
        fontsize=FONT_TITLE, loc="left", fontweight="bold"
    )
    n_ok = df_plot["z500_dam"].notna().sum()
    plt.text(
        0.98, 1.02, f"{n_ok} stations",
        ha="right", va="bottom", fontsize=FONT_COUNT,
        transform=ax.transAxes, 
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.to_datetime("now", utc=True).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"europe_500hpa_stations_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()

    return str(out_path.resolve())

# [Upper-air maps]

def plot_850hpa_gph_temperature_wind(
    ds: xr.Dataset,
    gph_interval: float = GPH_INTERVAL_850,
    isotherm_interval: float = ISOTHERM_INTERVAL_850,
    sigma: float = SIGMA_UPPER,
    output_dir: str = "../outputs",
) -> str:
    """
    850 hPa geopotential height, temperature, and wind.

    Parameters
    ----------
    ds : xr.Dataset
        GFS analysis dataset from ``fetch_gfs_analysis()``.
    gph_interval : float
        GPH contour interval (dam).
    isotherm_interval : float
        Temperature fill interval (°C).
    sigma : float
        Gaussian smoothing sigma applied to GPH and temperature fields.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    valid_time = ds.attrs.get("valid_time", "unknown")
    lon        = ds["longitude"].values
    lat        = ds["latitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    gph_dam = _smooth_field(ds["gh_850"].values / 10.0, sigma)
    temp_c  = _smooth_field(ds["t_850"].values - 273.15, sigma)
    u_raw   = ds["u_850"].values
    v_raw   = ds["v_850"].values

    fig, ax, proj, data_crs = _setup_europe_map(proj=ccrs.PlateCarree())

    # ── Temperature fill ────────────────────────────────────────────────────
    t_levs = np.arange(TEMP_MIN_850, TEMP_MAX_850 + isotherm_interval, isotherm_interval)
    cf = ax.contourf(
        lon_2d, lat_2d, temp_c,
        levels=t_levs, cmap="RdBu_r",
        transform=data_crs, extend="both",
    )
    _fs = max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD, axes_class=plt.Axes)
    cb = fig.colorbar(cf, cax=cax)
    cb.set_ticks(np.arange(TEMP_MIN_850, TEMP_MAX_850 + isotherm_interval, CBAR_TEMP_TICK_850))
    cb.set_label("[ °C ]", fontsize=_fs)
    cb.ax.tick_params(labelsize=_fs)

    # ── GPH contours (black) ────────────────────────────────────────────────
    g_min  = np.floor(np.nanmin(gph_dam) / gph_interval) * gph_interval
    g_max  = np.ceil( np.nanmax(gph_dam) / gph_interval) * gph_interval
    g_levs = np.arange(g_min, g_max + gph_interval, gph_interval)
    cs_g   = ax.contour(
        lon_2d, lat_2d, gph_dam,
        levels=g_levs, colors="black", linewidths=GPH_LINEWIDTH_850,
        transform=data_crs,
    )
    ax.clabel(
        cs_g, cs_g.levels[::GPH_LABEL_STRIDE_850], fmt="%d",
        fontsize=max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR)), inline=True,
    )

    # ── Wind barbs ──────────────────────────────────────────────────────────
    st = _BARB_STRIDE
    ax.barbs(
        lon_2d[::st, ::st], lat_2d[::st, ::st],
        u_raw[::st, ::st], v_raw[::st, ::st],
        length=BARB_LENGTH * BARB_SCALE_UPPER,
        transform=data_crs, color="black",
    )

    ax.set_title(
        _gfs_title("850 hPa Temperature [fill, °C], Geopotential Height [contours, dam], and Wind [barbs]", valid_time),
        fontsize=max(1, round(FONT_TITLE * FONT_SCALE_UPPER_TITLE)), loc="left", fontweight="bold",
    )

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.to_datetime("now", utc=True).strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"850hpa_gph_temp_wind_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    return str(out_path.resolve())


def plot_850hpa_temperature_advection(
    ds: xr.Dataset,
    T_adv: xr.DataArray,
    gph_interval: float = GPH_INTERVAL_850,
    sigma: float = SIGMA_UPPER,
    output_dir: str = "../outputs",
) -> str:
    """
    850 hPa temperature advection shading with geopotential height and wind barbs.

    Parameters
    ----------
    ds : xr.Dataset
        GFS analysis dataset.
    T_adv : xr.DataArray
        Temperature advection (K/s) from ``compute_temperature_advection()``.
    gph_interval : float
        GPH contour interval (dam).
    sigma : float
        Gaussian smoothing sigma applied to GPH.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    valid_time = ds.attrs.get("valid_time", "unknown")
    lon        = ds["longitude"].values
    lat        = ds["latitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    gph_dam    = _smooth_field(ds["gh_850"].values / 10.0, sigma)
    adv_vals   = np.asarray(T_adv.values, dtype=float)
    adv_scaled = adv_vals * TEMP_ADV_TIME_SCALE

    fig, ax, proj, data_crs = _setup_europe_map(proj=ccrs.PlateCarree())

    # ── Temperature advection fill ───────────────────────────────────────────────
    cf_levs = np.arange(TEMP_ADV_MIN_850, TEMP_ADV_MAX_850 + TEMP_ADV_INTERVAL_850, TEMP_ADV_INTERVAL_850)
    cf = ax.contourf(
        lon_2d, lat_2d, adv_scaled,
        levels=cf_levs, cmap="RdBu_r",
        transform=data_crs, extend="both",
    )
    _fs = max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD, axes_class=plt.Axes)
    cb = fig.colorbar(cf, cax=cax)
    cb.set_ticks(np.arange(TEMP_ADV_MIN_850, TEMP_ADV_MAX_850 + TEMP_ADV_INTERVAL_850, CBAR_TEMP_ADV_TICK_850))
    cb.set_label("[ °C / h ]", fontsize=_fs) # K per 1 h scaled to °C per 1 h
    cb.ax.tick_params(labelsize=_fs)

    # ── GPH contours ────────────────────────────────────────────────────────
    g_min  = np.floor(np.nanmin(gph_dam) / gph_interval) * gph_interval
    g_max  = np.ceil( np.nanmax(gph_dam) / gph_interval) * gph_interval
    g_levs = np.arange(g_min, g_max + gph_interval, gph_interval)
    cs_g   = ax.contour(
        lon_2d, lat_2d, gph_dam,
        levels=g_levs, colors="black", linewidths=GPH_LINEWIDTH_UPPER,
        transform=data_crs,
    )
    ax.clabel(
        cs_g, cs_g.levels[::GPH_LABEL_STRIDE], fmt="%d",
        fontsize=max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR)), inline=True,
    )

    # ── Wind barbs ──────────────────────────────────────────────────────────
    st = _BARB_STRIDE
    ax.barbs(
        lon_2d[::st, ::st], lat_2d[::st, ::st],
        ds["u_850"].values[::st, ::st], ds["v_850"].values[::st, ::st],
        length=BARB_LENGTH * BARB_SCALE_UPPER,
        transform=data_crs, color="black",
    )

    ax.set_title(
        _gfs_title("850 hPa Temperature Advection [fill, °C / h], Geopotential Height [contours, dam], and Wind [barbs]", valid_time),
        fontsize=max(1, round(FONT_TITLE * FONT_SCALE_UPPER_TITLE)), loc="left", fontweight="bold",
    )

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.to_datetime("now", utc=True).strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"850hpa_temperature_advection_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    return str(out_path.resolve())


def plot_700hpa_relative_humidity(
    ds: xr.Dataset,
    gph_interval: float = GPH_INTERVAL_700,
    sigma: float = SIGMA_UPPER,
    output_dir: str = "../outputs",
) -> str:
    """
    700 hPa relative humidity shading with geopotential height contours.

    Parameters
    ----------
    ds : xr.Dataset
        GFS analysis dataset.
    gph_interval : float
        GPH contour interval (dam).
    sigma : float
        Gaussian smoothing sigma applied to GPH and RH fields.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    valid_time = ds.attrs.get("valid_time", "unknown")
    lon        = ds["longitude"].values
    lat        = ds["latitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    gph_dam = _smooth_field(ds["gh_700"].values / 10.0, sigma)
    rh_vals = _smooth_field(ds["rh_700"].values, sigma)

    fig, ax, proj, data_crs = _setup_europe_map(proj=ccrs.PlateCarree())

    # ── RH fill ─────────────────────────────────────────────────────────────
    _n    = len(RH_CONTOUR_LEVELS_700) - 1
    _cmap = ListedColormap([plt.cm.BuGn(v) for v in (0.50, 0.70, 0.90)][:_n])
    rh_norm = BoundaryNorm(RH_CONTOUR_LEVELS_700, ncolors=_n)
    cf = ax.contourf(
        lon_2d, lat_2d, rh_vals,
        levels=RH_CONTOUR_LEVELS_700, cmap=_cmap, norm=rh_norm,
        transform=data_crs, extend="neither",
    )
    _fs = max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD, axes_class=plt.Axes)
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label("[ % ]", fontsize=_fs)
    cb.set_ticks(RH_CONTOUR_LEVELS_700)
    cb.ax.tick_params(labelsize=_fs)

    # ── GPH contours (blue) ─────────────────────────────────────────────────
    g_min  = np.floor(np.nanmin(gph_dam) / gph_interval) * gph_interval
    g_max  = np.ceil( np.nanmax(gph_dam) / gph_interval) * gph_interval
    g_levs = np.arange(g_min, g_max + gph_interval, gph_interval)
    cs_g   = ax.contour(
        lon_2d, lat_2d, gph_dam,
        levels=g_levs, colors="steelblue", linewidths=GPH_LINEWIDTH_UPPER,
        transform=data_crs,
    )
    ax.clabel(
        cs_g, cs_g.levels[::GPH_LABEL_STRIDE], fmt="%d",
        fontsize=max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR)), inline=True,
    )

    ax.set_title(
        _gfs_title("700 hPa Relative Humidity [fill, %], Geopotential Height [contours, dam]", valid_time),
        fontsize=max(1, round(FONT_TITLE * FONT_SCALE_UPPER_TITLE)), loc="left", fontweight="bold",
    )

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.to_datetime("now", utc=True).strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"700hpa_relative_humidity_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    return str(out_path.resolve())


def plot_500hpa_gph(
    ds: xr.Dataset,
    gph_interval: float = GPH_INTERVAL_500,
    sigma: float = SIGMA_UPPER,
    output_dir: str = "../outputs",
) -> str:
    """
    500 hPa geopotential height contours with wind barbs.

    Parameters
    ----------
    ds : xr.Dataset
        GFS analysis dataset.
    gph_interval : float
        GPH contour interval (dam).
    sigma : float
        Gaussian smoothing sigma applied to GPH.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    valid_time = ds.attrs.get("valid_time", "unknown")
    lon        = ds["longitude"].values
    lat        = ds["latitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    gph_dam = _smooth_field(ds["gh_500"].values / 10.0, sigma)

    fig, ax, proj, data_crs = _setup_europe_map(proj=ccrs.PlateCarree())

    g_min  = np.floor(np.nanmin(gph_dam) / gph_interval) * gph_interval
    g_max  = np.ceil( np.nanmax(gph_dam) / gph_interval) * gph_interval
    g_levs = np.arange(g_min, g_max + gph_interval, gph_interval)
    cs_g   = ax.contour(
        lon_2d, lat_2d, gph_dam,
        levels=g_levs, colors="black", linewidths=GPH_LINEWIDTH_500,
        transform=data_crs,
    )
    ax.clabel(
        cs_g, cs_g.levels[::GPH_LABEL_STRIDE_500], fmt="%d",
        fontsize=max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR)), inline=True,
    )

    ax.set_title(
        _gfs_title("500 hPa Geopotential Height [contours, dam]", valid_time),
        fontsize=max(1, round(FONT_TITLE * FONT_SCALE_UPPER_TITLE)), loc="left", fontweight="bold",
    )

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.to_datetime("now", utc=True).strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"500hpa_gph_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    return str(out_path.resolve())


def plot_500hpa_relative_vorticity(
    ds: xr.Dataset,
    rvort: xr.DataArray,
    gph_interval: float = GPH_INTERVAL_500,
    sigma: float = SIGMA_UPPER,
    output_dir: str = "../outputs",
) -> str:
    """
    500 hPa relative vorticity (fill), geopotential height (contours), and
    wind barbs.

    Parameters
    ----------
    ds : xr.Dataset
        GFS analysis dataset containing ``gh_500``, ``u_500``, ``v_500``.
    rvort : xr.DataArray
        Relative vorticity (1/s) computed from the full wind field.
    gph_interval : float
        GPH contour interval (dam).
    sigma : float
        Gaussian smoothing sigma applied to GPH.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    valid_time = ds.attrs.get("valid_time", "unknown")
    lon        = ds["longitude"].values
    lat        = ds["latitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    gph_dam   = _smooth_field(ds["gh_500"].values / 10.0, sigma)
    zeta_disp = np.asarray(rvort.values, dtype=float) * VORTICITY_DISPLAY_SCALE

    fig, ax, proj, data_crs = _setup_europe_map(proj=ccrs.PlateCarree())
    fig.patch.set_facecolor("w")

    # ── Vorticity fill ──────────────────────────────────────────────────────
    z_levs = np.arange(-VORT_MAX_500, VORT_MAX_500 + VORT_INTERVAL_500, VORT_INTERVAL_500)
    cf = ax.contourf(
        lon_2d, lat_2d, zeta_disp,
        levels=z_levs, cmap=_pvort_cmap,
        transform=data_crs, extend="both",
    )
    _fs = max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD, axes_class=plt.Axes)
    cb = fig.colorbar(cf, cax=cax)
    cb.set_ticks(np.arange(-VORT_MAX_500, VORT_MAX_500 + VORT_INTERVAL_500, CBAR_VORT_TICK_500))
    cb.set_label(r"[ ×10⁻⁵ s⁻¹ ]", fontsize=_fs)
    cb.ax.tick_params(labelsize=_fs)

    # ── GPH contours ────────────────────────────────────────────────────────
    g_min  = np.floor(np.nanmin(gph_dam) / gph_interval) * gph_interval
    g_maxv = np.ceil( np.nanmax(gph_dam) / gph_interval) * gph_interval
    g_levs = np.arange(g_min, g_maxv + gph_interval, gph_interval)
    _fl = max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR))
    cs = ax.contour(
        lon_2d, lat_2d, gph_dam,
        levels=g_levs, colors="black", linewidths=GPH_LINEWIDTH_500,
        transform=data_crs,
    )
    ax.clabel(cs, cs.levels[::GPH_LABEL_STRIDE_500], fmt="%d", fontsize=_fl, inline=True)

    # ── Wind barbs ──────────────────────────────────────────────────────────
    st = _BARB_STRIDE
    ax.barbs(
        lon_2d[::st, ::st], lat_2d[::st, ::st],
        ds["u_500"].values[::st, ::st], ds["v_500"].values[::st, ::st],
        length=BARB_LENGTH * BARB_SCALE_UPPER,
        transform=data_crs, color="black", linewidth=0.6,
    )

    ax.set_title(
        _gfs_title(
            r"500 hPa Relative Vorticity [fill, ×10⁻⁵ s⁻¹], Geopotential Height [contours, dam], Wind",
            valid_time,
        ),
        fontsize=max(1, round(FONT_TITLE * FONT_SCALE_UPPER_TITLE)),
        loc="left", fontweight="bold",
    )

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.to_datetime("now", utc=True).strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"500hpa_vorticity_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    return str(out_path.resolve())


def plot_500hpa_relative_vorticity_advection(
    ds: xr.Dataset,
    rvort_adv: xr.DataArray,
    gph_interval: float = GPH_INTERVAL_500,
    sigma: float = SIGMA_UPPER,
    output_dir: str = "../outputs",
) -> str:
    """
    500 hPa relative vorticity advection (fill, PVA red / NVA blue),
    geopotential height (contours), and wind barbs.

    Parameters
    ----------
    ds : xr.Dataset
        GFS analysis dataset containing ``gh_500``, ``u_500``, ``v_500``.
    rvort_adv : xr.DataArray
        Relative vorticity advection (1/s²) from
        ``compute_relative_vorticity_advection()``.
    gph_interval : float
        GPH contour interval (dam).
    sigma : float
        Gaussian smoothing sigma applied to GPH.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    valid_time = ds.attrs.get("valid_time", "unknown")
    lon        = ds["longitude"].values
    lat        = ds["latitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    gph_dam   = _smooth_field(ds["gh_500"].values / 10.0, sigma)
    zadv_disp = np.asarray(rvort_adv.values, dtype=float) * 1e9
    a_max     = max(np.nanpercentile(np.abs(zadv_disp), 97), 0.1)

    fig, ax, proj, data_crs = _setup_europe_map(proj=ccrs.PlateCarree())
    fig.patch.set_facecolor("w")

    # ── Vorticity advection fill ─────────────────────────────────────────────
    a_levs = np.linspace(-a_max, a_max, 21)
    cf = ax.contourf(
        lon_2d, lat_2d, zadv_disp,
        levels=a_levs, cmap="RdBu_r",
        transform=data_crs, extend="both",
    )
    _fs = max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD, axes_class=plt.Axes)
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label(r"[ ×10⁻⁹ s⁻² ]", fontsize=_fs)
    cb.ax.tick_params(labelsize=_fs)

    # ── GPH contours ─────────────────────────────────────────────────────────
    g_min  = np.floor(np.nanmin(gph_dam) / gph_interval) * gph_interval
    g_maxv = np.ceil( np.nanmax(gph_dam) / gph_interval) * gph_interval
    g_levs = np.arange(g_min, g_maxv + gph_interval, gph_interval)
    _fl = max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR))
    cs = ax.contour(
        lon_2d, lat_2d, gph_dam,
        levels=g_levs, colors="black", linewidths=GPH_LINEWIDTH_500,
        transform=data_crs,
    )
    ax.clabel(cs, cs.levels[::GPH_LABEL_STRIDE_500], fmt="%d", fontsize=_fl, inline=True)

    # ── Wind barbs ───────────────────────────────────────────────────────────
    st = _BARB_STRIDE
    ax.barbs(
        lon_2d[::st, ::st], lat_2d[::st, ::st],
        ds["u_500"].values[::st, ::st], ds["v_500"].values[::st, ::st],
        length=BARB_LENGTH * BARB_SCALE_UPPER,
        transform=data_crs, color="black", linewidth=0.6,
    )

    ax.set_title(
        _gfs_title(
            r"500 hPa Relative Vorticity Advection [fill, ×10⁻⁹ s⁻², PVA red / NVA blue],"
            " Geopotential Height [contours, dam], Wind",
            valid_time,
        ),
        fontsize=max(1, round(FONT_TITLE * FONT_SCALE_UPPER_TITLE)),
        loc="left", fontweight="bold",
    )

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.to_datetime("now", utc=True).strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"500hpa_vorticity_advection_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    return str(out_path.resolve())


def plot_250hpa_jet(
    ds: xr.Dataset,
    wspd: xr.DataArray,
    gph_interval: float = GPH_INTERVAL_250,
    isotach_interval: int = ISOTACH_INTERVAL,
    sigma: float = SIGMA_UPPER,
    output_dir: str = "../outputs",
) -> str:
    """
    250 hPa wind speed (isotachs), geopotential height, and wind barbs.

    Parameters
    ----------
    ds : xr.Dataset
        GFS analysis dataset.
    wspd : xr.DataArray
        Wind speed (m/s) from ``compute_wind_speed()``.
    gph_interval : float
        GPH contour interval (dam).
    isotach_interval : int
        Isotach fill interval (m/s).
    sigma : float
        Gaussian smoothing sigma applied to GPH.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    valid_time = ds.attrs.get("valid_time", "unknown")
    lon        = ds["longitude"].values
    lat        = ds["latitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    gph_dam   = _smooth_field(ds["gh_250"].values / 10.0, sigma)
    wspd_vals = np.asarray(wspd.values, dtype=float)

    fig, ax, proj, data_crs = _setup_europe_map(proj=ccrs.PlateCarree())

    # ── Isotach fill ────────────────────────────────────────────────────────
    iso_levs = list(range(ISOTACH_MIN, ISOTACH_MAX, isotach_interval))
    cf = ax.contourf(
        lon_2d, lat_2d, wspd_vals,
        levels=iso_levs, cmap="YlOrRd",
        transform=data_crs, extend="max",
    )
    _fs = max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD, axes_class=plt.Axes)
    cb = fig.colorbar(cf, cax=cax)
    cb.set_ticks(range(ISOTACH_MIN, ISOTACH_MAX, CBAR_ISOTACH_TICK_250))
    cb.set_label("[ m/s ]", fontsize=_fs)
    cb.ax.tick_params(labelsize=_fs)

    # ── Jet-core contour (emphasized) ───────────────────────────────────────
    ax.contour(
        lon_2d, lat_2d, wspd_vals,
        levels=[JET_CORE_LEVEL], colors=JET_CORE_COLOR, linewidths=JET_CORE_LINEWIDTH,
        transform=data_crs,
    )

    # ── GPH contours ────────────────────────────────────────────────────────
    g_min  = np.floor(np.nanmin(gph_dam) / gph_interval) * gph_interval
    g_max  = np.ceil( np.nanmax(gph_dam) / gph_interval) * gph_interval
    g_levs = np.arange(g_min, g_max + gph_interval, gph_interval)
    cs_g   = ax.contour(
        lon_2d, lat_2d, gph_dam,
        levels=g_levs, colors="black", linewidths=GPH_LINEWIDTH_UPPER,
        transform=data_crs,
    )
    ax.clabel(
        cs_g, cs_g.levels[::GPH_LABEL_STRIDE_250], fmt="%d",
        fontsize=max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR)), inline=True,
    )

    # ── Wind barbs ──────────────────────────────────────────────────────────
    st = _BARB_STRIDE
    ax.barbs(
        lon_2d[::st, ::st], lat_2d[::st, ::st],
        ds["u_250"].values[::st, ::st], ds["v_250"].values[::st, ::st],
        length=BARB_LENGTH * BARB_SCALE_UPPER,
        transform=data_crs, color="black",
    )

    ax.set_title(
        _gfs_title("250 hPa Wind Speed [fill, m/s], Geopotential Height [contours, dam], and Wind [barbs]", valid_time),
        fontsize=max(1, round(FONT_TITLE * FONT_SCALE_UPPER_TITLE)), loc="left", fontweight="bold",
    )

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.to_datetime("now", utc=True).strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"250hpa_jet_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    return str(out_path.resolve())


def plot_upper_air_overview(
    ds: xr.Dataset,
    T_adv: xr.DataArray,
    rvort: xr.DataArray,
    wspd: xr.DataArray,
    sigma: float = SIGMA_UPPER,
    output_dir: str = "../outputs",
) -> str:
    """
    Four-panel upper-air synthesis figure.

    Assembles the four upper-air diagnostics at the common analysis valid
    time into a 2 × 2 figure:

    * Upper-left  : 850 hPa temperature advection + GPH
    * Upper-right : 700 hPa relative humidity + GPH
    * Lower-left  : 500 hPa relative vorticity + GPH
    * Lower-right : 250 hPa isotachs + GPH + wind barbs

    Parameters
    ----------
    ds : xr.Dataset
        GFS analysis dataset.
    T_adv : xr.DataArray
        Temperature advection (K/s) from ``compute_temperature_advection()``.
    rvort : xr.DataArray
        Relative vorticity (1/s) computed from the full wind field.
    wspd : xr.DataArray
        Wind speed at 250 hPa (m/s) from ``compute_wind_speed()``.
    sigma : float
        Gaussian smoothing sigma applied to all gridded fields.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    valid_time = ds.attrs.get("valid_time", "unknown")
    lon        = ds["longitude"].values
    lat        = ds["latitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    gph_850  = _smooth_field(ds["gh_850"].values / 10.0, sigma)
    gph_700  = _smooth_field(ds["gh_700"].values / 10.0, sigma)
    gph_500  = _smooth_field(ds["gh_500"].values / 10.0, sigma)
    gph_250  = _smooth_field(ds["gh_250"].values / 10.0, sigma)
    rh_700   = _smooth_field(ds["rh_700"].values, sigma)
    wspd_arr = np.asarray(wspd.values, dtype=float)

    adv_scaled = np.asarray(T_adv.values,  dtype=float) * TEMP_ADV_TIME_SCALE
    zeta_disp  = np.asarray(rvort.values, dtype=float) * VORTICITY_DISPLAY_SCALE

    proj     = ccrs.PlateCarree()
    data_crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2, 2,
        figsize=(FIG_SIZE_IN[0] * FIG_SIZE_SCALE_OVERVIEW, FIG_HEIGHT_OVERVIEW),
        subplot_kw={"projection": proj},
        gridspec_kw={"wspace": SUBPLOT_WSPACE_OVERVIEW, "hspace": SUBPLOT_HSPACE_OVERVIEW},
    )
    fig.patch.set_facecolor("w")

    ax_tl, ax_tr = axes[0, 0], axes[0, 1]
    ax_bl, ax_br = axes[1, 0], axes[1, 1]
    for ax in axes.flat:
        _setup_europe_map(ax=ax)

    _ft = max(1, round(FONT_TITLE * FONT_SCALE_TITLE_OVERVIEW))
    _fl = max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR * FONT_SCALE_CONTOUR_OVERVIEW))
    _fs = max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR * FONT_SCALE_TITLE_OVERVIEW))
    st2 = _BARB_STRIDE * BARB_STRIDE_SCALE_OVERVIEW
    _bl = BARB_LENGTH * BARB_SCALE_STATION * BARB_SCALE_OVERVIEW

    def _gph_cs(ax, gph_dam, interval, color="black"):
        g_min  = np.floor(np.nanmin(gph_dam) / interval) * interval
        g_max  = np.ceil( np.nanmax(gph_dam) / interval) * interval
        g_levs = np.arange(g_min, g_max + interval, interval)
        cs     = ax.contour(lon_2d, lat_2d, gph_dam, levels=g_levs,
                            colors=color, linewidths=0.8, transform=data_crs)
        ax.clabel(cs, cs.levels[::GPH_LABEL_STRIDE], fmt="%d", fontsize=_fl, inline=True)

    def _cbar(ax, cf, label, ticks=None):
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD, axes_class=plt.Axes)
        cb  = fig.colorbar(cf, cax=cax)
        cb.set_label(label, fontsize=_fs)
        if ticks is not None:
            cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=_fs)

    # ── TL: 850 hPa temperature advection ───────────────────────────────────
    cf_tl = ax_tl.contourf(
        lon_2d, lat_2d, adv_scaled,
        levels=np.arange(TEMP_ADV_MIN_850, TEMP_ADV_MAX_850 + TEMP_ADV_INTERVAL_850, TEMP_ADV_INTERVAL_850),
        cmap="RdBu_r", transform=data_crs, extend="both",
    )
    _cbar(ax_tl, cf_tl, "[ °C / h ]",
          ticks=np.arange(TEMP_ADV_MIN_850, TEMP_ADV_MAX_850 + TEMP_ADV_INTERVAL_850, CBAR_TEMP_ADV_TICK_850))
    _gph_cs(ax_tl, gph_850, GPH_INTERVAL_850)
    ax_tl.barbs(
        lon_2d[::st2, ::st2], lat_2d[::st2, ::st2],
        ds["u_850"].values[::st2, ::st2], ds["v_850"].values[::st2, ::st2],
        length=_bl, transform=data_crs, color="black",
    )
    ax_tl.set_title("850 hPa Temperature Advection [fill, °C / h],\nGeopotential Height [contours, dam], Wind [barbs]", fontsize=_ft, loc="left", fontweight="bold")

    # ── TR: 700 hPa relative humidity ───────────────────────────────────────
    _n_rh   = len(RH_CONTOUR_LEVELS_700) - 1
    _rh_cmap = ListedColormap([plt.cm.BuGn(v) for v in (0.50, 0.70, 0.90)][:_n_rh])
    rh_norm = BoundaryNorm(RH_CONTOUR_LEVELS_700, ncolors=_n_rh)
    cf_tr = ax_tr.contourf(
        lon_2d, lat_2d, rh_700,
        levels=RH_CONTOUR_LEVELS_700, cmap=_rh_cmap, norm=rh_norm,
        transform=data_crs, extend="neither",
    )
    _cbar(ax_tr, cf_tr, "[ % ]", ticks=RH_CONTOUR_LEVELS_700)
    _gph_cs(ax_tr, gph_700, GPH_INTERVAL_700, color="steelblue")
    ax_tr.set_title("700 hPa Relative Humidity [fill, %],\nGeopotential Height [contours, dam]", fontsize=_ft, loc="left", fontweight="bold")

    # ── BL: 500 hPa relative vorticity ──────────────────────────────────────
    cf_bl = ax_bl.contourf(
        lon_2d, lat_2d, zeta_disp,
        levels=np.arange(-VORT_MAX_500, VORT_MAX_500 + VORT_INTERVAL_500, VORT_INTERVAL_500),
        cmap=_pvort_cmap, transform=data_crs, extend="both",
    )
    _cbar(ax_bl, cf_bl, r"[ ×10⁻⁵ s⁻¹ ]",
          ticks=np.arange(-VORT_MAX_500, VORT_MAX_500 + VORT_INTERVAL_500, CBAR_VORT_TICK_500))
    _gph_cs(ax_bl, gph_500, GPH_INTERVAL_500)
    ax_bl.barbs(
        lon_2d[::st2, ::st2], lat_2d[::st2, ::st2],
        ds["u_500"].values[::st2, ::st2], ds["v_500"].values[::st2, ::st2],
        length=_bl, transform=data_crs, color="black",
    )
    ax_bl.set_title("500 hPa Relative Vorticity [fill, ×10⁻⁵ s⁻¹],\nGeopotential Height [contours, dam], Wind [barbs]", fontsize=_ft, loc="left", fontweight="bold")

    # ── BR: 250 hPa isotachs + barbs ────────────────────────────────────────
    iso_levs = list(range(ISOTACH_MIN, ISOTACH_MAX, ISOTACH_INTERVAL))
    cf_br = ax_br.contourf(
        lon_2d, lat_2d, wspd_arr,
        levels=iso_levs, cmap="YlOrRd", transform=data_crs, extend="max",
    )
    _cbar(ax_br, cf_br, "[ m/s ]", ticks=list(range(ISOTACH_MIN, ISOTACH_MAX, CBAR_ISOTACH_TICK_250)))
    ax_br.contour(lon_2d, lat_2d, wspd_arr, levels=[JET_CORE_LEVEL],
                  colors=JET_CORE_COLOR, linewidths=JET_CORE_LINEWIDTH * 0.8, transform=data_crs)
    _gph_cs(ax_br, gph_250, GPH_INTERVAL_250)
    ax_br.barbs(
        lon_2d[::st2, ::st2], lat_2d[::st2, ::st2],
        ds["u_250"].values[::st2, ::st2], ds["v_250"].values[::st2, ::st2],
        length=_bl, transform=data_crs, color="black",
    )
    ax_br.set_title("250 hPa Wind Speed [fill, m/s],\nGeopotential Height [contours, dam], and Wind [barbs]", fontsize=_ft, loc="left", fontweight="bold")

    fig.suptitle(
        _gfs_title("Upper-Air Synoptic Overview", valid_time),
        fontsize=max(1, round(FONT_TITLE * FONT_SCALE_SUPTITLE_OVERVIEW)), fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(top=0.93)

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.to_datetime("now", utc=True).strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"upper_air_overview_{timestamp}.png"
    fig.savefig(out_path, dpi=FIG_DPI_OVERVIEW, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    return str(out_path.resolve())


def plot_four_panel_forecast_animation(
    dataset: xr.Dataset,
    domain: tuple[float, float, float, float],
    output_dir: str | Path,
    dpi: int = FIG_DPI_OVERVIEW,
) -> Path:
    """
    Build an animated four-panel upper-air chart from a GFS forecast Dataset.

    One frame per lead time. Panel layout:

    - top-left    : 850 hPa temperature shaded + GPH contours + wind barbs
    - top-right   : 700 hPa relative humidity shaded + GPH contours
    - bottom-left : 500 hPa GPH contoured with wind barbs
    - bottom-right: 250 hPa wind speed shaded + GPH + jet-core contour + barbs

    Parameters
    ----------
    dataset : xr.Dataset
        GFS forecast Dataset from ``fetch_gfs_forecast()``.
        Required variables: ``t_850``, ``gh_850``, ``u_850``, ``v_850``,
        ``rh_700``, ``gh_700``, ``gh_500``, ``u_500``, ``v_500``,
        ``u_250``, ``v_250``, ``gh_250``.
    domain : tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees. The map extent
        is controlled by ``_setup_europe_map`` (``EUROPE_EXTENT``); this
        parameter is retained for API compatibility.
    output_dir : str or Path
        Directory for the exported GIF. Created if absent.
    dpi : int
        Output resolution (dots per inch). Defaults to ``FIG_DPI_OVERVIEW``.

    Returns
    -------
    Path
        Absolute path to the saved animated GIF. Individual frame PNGs are
        also saved to ``output_dir`` as
        ``gfs_frame_<init_yyyymmdd_hhmm>_fxx<NNN>h.png``.
    """
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_crs   = ccrs.PlateCarree()
    lead_times = dataset.valid_time.values
    init_date  = dataset.attrs.get("init_date", "unknown")

    lon    = dataset.longitude.values
    lat    = dataset.latitude.values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    _ft = max(1, round(FONT_TITLE * FONT_SCALE_TITLE_OVERVIEW))
    _fl = max(1, round(FONT_LABEL * FONT_SCALE_CONTOUR * FONT_SCALE_CONTOUR_OVERVIEW))
    _fs = max(1, round(FONT_LABEL * FONT_SCALE_UPPER_CBAR * FONT_SCALE_TITLE_OVERVIEW))
    _st = _BARB_STRIDE * BARB_STRIDE_SCALE_OVERVIEW
    _bl = BARB_LENGTH * BARB_SCALE_STATION * BARB_SCALE_OVERVIEW

    def _smooth(arr: np.ndarray) -> np.ndarray:
        """NaN-safe Gaussian smooth via weighted convolution."""
        arr = arr.astype(float)
        valid = np.isfinite(arr)
        if valid.all():
            return gaussian_filter(arr, sigma=SIGMA_UPPER)
        if not valid.any():
            return arr.copy()
        filled  = np.where(valid, arr, 0.0)
        weights = valid.astype(float)
        return gaussian_filter(filled, sigma=SIGMA_UPPER) / np.maximum(
            gaussian_filter(weights, sigma=SIGMA_UPPER), 1e-12,
        )

    def _gph_cs(ax, gph_dam, interval, color="black"):
        if not np.isfinite(gph_dam).any():
            LOG.warning("_gph_cs: all-NaN GPH field — contour skipped")
            return
        g_min  = np.floor(np.nanmin(gph_dam) / interval) * interval
        g_max  = np.ceil( np.nanmax(gph_dam) / interval) * interval
        g_levs = np.arange(g_min, g_max + interval, interval)
        cs     = ax.contour(lon_2d, lat_2d, gph_dam, levels=g_levs,
                            colors=color, linewidths=0.8, transform=data_crs)
        ax.clabel(cs, cs.levels[::GPH_LABEL_STRIDE], fmt="%d", fontsize=_fl, inline=True)

    def _cbar_anim(ax, cf, label, ticks=None):
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD, axes_class=plt.Axes)
        cb  = fig.colorbar(cf, cax=cax)
        cb.set_label(label, fontsize=_fs)
        if ticks is not None:
            cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=_fs)

    def _draw_panels(axes, ds_step, lead_h):
        """Clear and redraw all four panels for one forecast lead time."""
        ax_tl, ax_tr, ax_bl, ax_br = axes
        for ax in axes:
            ax.cla()
            _setup_europe_map(ax=ax)

        gph_850 = _smooth(ds_step["gh_850"].values / 10.0)
        gph_700 = _smooth(ds_step["gh_700"].values / 10.0)
        gph_500 = _smooth(ds_step["gh_500"].values / 10.0)
        gph_250 = _smooth(ds_step["gh_250"].values / 10.0)

        cycle      = pd.Timestamp(init_date)
        valid_dt   = cycle + pd.Timedelta(hours=lead_h)
        time_label = (
            f"Init: {cycle.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Valid: {valid_dt.strftime('%Y-%m-%d %H:%M UTC')}"
        )

        # ── TL: 850 hPa temperature shaded + GPH contours + wind barbs ─────────
        t850   = _smooth(ds_step["t_850"].values - 273.15)
        t_levs = np.arange(TEMP_MIN_850, TEMP_MAX_850 + ISOTHERM_INTERVAL_850, ISOTHERM_INTERVAL_850)
        im_tl  = ax_tl.contourf(
            lon_2d, lat_2d, t850, levels=t_levs,
            cmap="RdYlBu_r", transform=data_crs, extend="both",
        )
        _gph_cs(ax_tl, gph_850, GPH_INTERVAL_850)
        ax_tl.barbs(
            lon_2d[::_st, ::_st], lat_2d[::_st, ::_st],
            ds_step["u_850"].values[::_st, ::_st], ds_step["v_850"].values[::_st, ::_st],
            length=_bl, transform=data_crs, color="black",
        )
        ax_tl.set_title(
            "850 hPa Temperature [fill, °C],\nGeopotential Height [contours, dam], Wind [barbs]",
            fontsize=_ft, loc="left", fontweight="bold",
        )
        ax_tl.set_title(time_label, fontsize=_ft, loc="right")

        # ── TR: 700 hPa RH shaded + GPH contours (steelblue) ──────────────────
        rh700 = _smooth(ds_step["rh_700"].values)
        im_tr = ax_tr.contourf(
            lon_2d, lat_2d, rh700, levels=RH_CONTOUR_LEVELS_700,
            cmap=_RH_CMAP, norm=_RH_NORM, transform=data_crs, extend="neither",
        )
        _gph_cs(ax_tr, gph_700, GPH_INTERVAL_700, color="steelblue")
        ax_tr.set_title(
            "700 hPa Relative Humidity [fill, %],\nGeopotential Height [contours, dam]",
            fontsize=_ft, loc="left", fontweight="bold",
        )
        ax_tr.set_title(time_label, fontsize=_ft, loc="right")

        # ── BL: 500 hPa GPH contoured + wind barbs ────────────────────────────
        _gph_cs(ax_bl, gph_500, GPH_INTERVAL_500)
        ax_bl.barbs(
            lon_2d[::_st, ::_st], lat_2d[::_st, ::_st],
            ds_step["u_500"].values[::_st, ::_st], ds_step["v_500"].values[::_st, ::_st],
            length=_bl, transform=data_crs, color="black",
        )
        ax_bl.set_title(
            "500 hPa Geopotential Height [contours, dam],\nWind [barbs]",
            fontsize=_ft, loc="left", fontweight="bold",
        )
        ax_bl.set_title(time_label, fontsize=_ft, loc="right")

        # ── BR: 250 hPa isotachs + jet core + GPH + barbs ─────────────────────
        spd250 = _smooth(
            np.sqrt(ds_step["u_250"].values**2 + ds_step["v_250"].values**2)
        )
        im_br = ax_br.contourf(
            lon_2d, lat_2d, spd250, levels=_ISO_LEVS,
            cmap="YlOrRd", transform=data_crs, extend="max",
        )
        ax_br.contour(
            lon_2d, lat_2d, spd250, levels=[JET_CORE_LEVEL],
            colors=JET_CORE_COLOR, linewidths=JET_CORE_LINEWIDTH * 0.8, transform=data_crs,
        )
        _gph_cs(ax_br, gph_250, GPH_INTERVAL_250)
        ax_br.barbs(
            lon_2d[::_st, ::_st], lat_2d[::_st, ::_st],
            ds_step["u_250"].values[::_st, ::_st], ds_step["v_250"].values[::_st, ::_st],
            length=_bl, transform=data_crs, color="black",
        )
        ax_br.set_title(
            "250 hPa Wind Speed [fill, m/s],\nGeopotential Height [contours, dam], Wind [barbs]",
            fontsize=_ft, loc="left", fontweight="bold",
        )
        ax_br.set_title(time_label, fontsize=_ft, loc="right")

        return im_tl, im_tr, im_br

    fig, axes_2d = plt.subplots(
        2, 2,
        figsize=(FIG_SIZE_IN[0] * FIG_SIZE_SCALE_OVERVIEW, FIG_HEIGHT_OVERVIEW),
        subplot_kw={"projection": data_crs},
        gridspec_kw={"wspace": SUBPLOT_WSPACE_OVERVIEW, "hspace": SUBPLOT_HSPACE_OVERVIEW_FORECAST},
    )
    fig.patch.set_facecolor("w")
    axes = axes_2d.flatten()

    lead_h_0 = int(lead_times[0])
    im_tl, im_tr, im_br = _draw_panels(axes, dataset.sel(valid_time=lead_h_0), lead_h_0)
    _cbar_anim(axes[0], im_tl, "[ °C ]",
               ticks=np.arange(TEMP_MIN_850, TEMP_MAX_850 + CBAR_TEMP_TICK_850, CBAR_TEMP_TICK_850))
    _cbar_anim(axes[1], im_tr, "[ % ]", ticks=RH_CONTOUR_LEVELS_700)
    _cbar_anim(axes[3], im_br, "[ m/s ]",
               ticks=list(range(ISOTACH_MIN, ISOTACH_MAX, CBAR_ISOTACH_TICK_250)))
    _div_bl = make_axes_locatable(axes[2])
    _cax_bl = _div_bl.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD, axes_class=plt.Axes)
    _cax_bl.set_visible(False)

    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M")
    out_path  = out_dir / f"gfs_four_panel_animation_{timestamp}.gif"
    cycle_ts  = pd.Timestamp(init_date)
    init_safe = cycle_ts.strftime("%Y%m%d_%H%M")
    gif_frames: list[Image] = []

    for lead_h in [int(lt) for lt in lead_times]:
        _draw_panels(axes, dataset.sel(valid_time=lead_h), lead_h)
        frame_path = out_dir / f"gfs_frame_{init_safe}_f{lead_h:03d}h.png"
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        buf.seek(0)
        gif_frames.append(Image.open(buf).copy())

    gif_frames[0].save(
        str(out_path),
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        duration=_GIF_INTERVAL_MS,
        loop=0,
        optimize=False,
    )
    plt.close(fig)

    return out_path.resolve()


# [Skew-T diagrams]

def init_skewt_figure(p_min, p_max, t_min, t_max) -> tuple:
    """
    Create a blank Skew-T log-P figure with isopleths and isotherms,
    and with axis limits applied.

    Parameters
    ----------
    p_min : float
        Top of pressure axis (hPa).
    p_max : float
        Bottom of pressure axis (hPa).
    t_min : float
        Left edge of temperature axis (°C).
    t_max : float
        Right edge of temperature axis (°C).

    Returns
    -------
    tuple[matplotlib.figure.Figure, metpy.plots.SkewT]
        The figure and the SkewT axes wrapper.
    """
    fig = plt.figure(figsize=SKEWT_FIG_SIZE)
    skew = SkewT(fig, rotation=SKEWT_ROTATION)
    skew.ax.set_ylim(p_max, p_min)
    skew.ax.set_xlim(t_min, t_max)
    skew.ax.set_xlabel("Temperature [ °C ]", fontsize=FONT_LABEL * FONT_SCALE_STATION) # font size is scalded down based on font_scale_station to avoid duplicates
    skew.ax.set_ylabel("Pressure [ hPa ]", fontsize=FONT_LABEL * FONT_SCALE_STATION) # font size is scalded down based on font_scale_station to avoid duplicates
    skew.ax.tick_params(labelsize=round(FONT_LABEL * FONT_SCALE_CONTOUR)) # font size scaled down via FONT_SCALE_CONTOUR to avoid clashing with cell title
    skew.ax.axvline(0, linestyle = SKEWT_ZERO_ISOTHERM_STYLE, 
                    color=SKEWT_ZERO_ISOTHERM_COLOR,
                    linewidth=SKEWT_ZERO_ISOTHERM_LINEWIDTH,
                    alpha=SKEWT_ZERO_ISOTHERM_ALPHA)
    return fig, skew


def add_skewt_mixing_lines(skew) -> None:
    """
    Draw constant-mixing-ratio lines with value labels at the bottom of the diagram.
    """
    w_lines = np.array(SKEWT_MIXING_RATIO_LEVELS) * units("g/kg")
    skew.plot_mixing_lines(
        mixing_ratio=w_lines,
        linestyle=SKEWT_MIXING_LINE_STYLE, color=SKEWT_MIXING_LINE_COLOR,
        alpha = SKEWT_MIXING_LINE_ALPHA, linewidth=SKEWT_MIXING_LINE_WIDTH,
    )
    p_bottom = skew.ax.get_ylim()[0]   # p_max — bottom of diagram
    t_min, t_max = skew.ax.get_xlim()
    for w in SKEWT_MIXING_RATIO_LEVELS:
        e = w * p_bottom / (622.0 + w)
        if e <= 0:
            continue
        try:
            T_label = 243.5 * math.log(e / 6.112) / (17.67 - math.log(e / 6.112))
        except ValueError:
            continue
        if not (t_min <= T_label <= t_max):
            continue
        label = f"{w:g}"
        skew.ax.text(
            T_label, p_bottom, label,
            fontsize=SKEWT_MIXING_LABEL_FONTSIZE, color=SKEWT_MIXING_LINE_COLOR,
            ha="center", va="top", clip_on=False,
        )


def add_skewt_dry_adiabats(skew, t_min, t_max) -> None:
    """
    Draw dry adiabats spanning t_min–t_max at interval defined by ``SKEWT_DRY_ADIABAT_*``.
    """
    t0_vals = np.arange(t_min, t_max + 1, SKEWT_DRY_ADIABAT_INTERVAL) * units.degC
    skew.plot_dry_adiabats(
        t0=t0_vals,
        linestyle=SKEWT_DRY_ADIABAT_STYLE, color=SKEWT_DRY_ADIABAT_COLOR,
        alpha = SKEWT_DRY_ADIABAT_ALPHA, linewidth=SKEWT_DRY_ADIABAT_LINEWIDTH,
    )


def add_skewt_moist_adiabats(skew, t_min, t_max) -> None:
    """
    Draw moist adiabats spanning t_min–t_max at interval defined by ``SKEWT_MOIST_ADIABAT_*``.
    """
    t0_vals = np.arange(t_min, t_max + 1, SKEWT_MOIST_ADIABAT_INTERVAL) * units.degC
    skew.plot_moist_adiabats(
        t0=t0_vals,
        linestyle=SKEWT_MOIST_ADIABAT_STYLE, color=SKEWT_MOIST_ADIABAT_COLOR,
        alpha = SKEWT_MOIST_ADIABAT_ALPHA, linewidth=SKEWT_MOIST_ADIABAT_LINEWIDTH,
    )

def plot_skewt_sounding(p, T, Td, p_min, p_max, t_min, t_max, *,
                        station_id=None, valid_dt=None,
                        output_dir=None) -> tuple:
    """
    Overlay the radiosonde temperature and dew-point profiles on the Skew-T
    background plot, and save the figure.

    Parameters
    ----------
    p : pint.Quantity
        Pressure profile in hPa, shape (N,), monotonically decreasing.
    T : pint.Quantity
        Temperature profile in °C, shape (N,).
    Td : pint.Quantity
        Dew-point profile in °C, shape (N,).
    p_min, p_max : float
        Pressure axis extents in hPa (top and bottom).
    t_min, t_max : float
        Temperature axis extents in °C (left and right edges).
    station_id : str or None
        WMO/ICAO station identifier shown in the upper-left corner.
    valid_dt : datetime or None
        Sounding valid time shown in the upper-right corner.
    output_dir : str or None
        Directory to save the figure. If None, the figure is not saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, metpy.plots.SkewT]
        The figure and the SkewT axes wrapper with the sounding overlaid.
    """
    fig, skew = init_skewt_figure(p_min, p_max, t_min, t_max)
    add_skewt_mixing_lines(skew)
    add_skewt_dry_adiabats(skew, t_min, t_max)
    add_skewt_moist_adiabats(skew, t_min, t_max)
    skew.plot(p, T, "r", linewidth=SKEWT_PROFILE_LINEWIDTH, label="Temperature")
    skew.plot(p, Td, "g", linewidth=SKEWT_PROFILE_LINEWIDTH, label="Dew Point")
    skew.ax.legend(loc="upper right", fontsize=round(FONT_LABEL * FONT_SCALE_CONTOUR))
    if station_id is not None or valid_dt is not None:
        title_y = skew.ax.get_position().y1 + 0.01
        if station_id is not None:
            fig.text(0.12, title_y, f"{station_id} Sounding",
                     fontsize=FONT_TITLE * FONT_SCALE_CONTOUR, fontweight="bold", # font size scaled down via FONT_SCALE_CONTOUR to avoid clashing with cell title
                     va="bottom", ha="left")
        if valid_dt is not None:
            fig.text(0.90, title_y, valid_dt.strftime("Valid time: %Y-%m-%d %H:%M UTC"),
                     fontsize=FONT_TITLE * FONT_SCALE_CONTOUR, # font size scaled down via FONT_SCALE_CONTOUR to avoid clashing with cell title
                     va="bottom", ha="right")
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        fig.savefig(out_dir / f"skewt_sounding_{timestamp}.png",
                    dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    return fig, skew


def plot_skewt_parcel_diagnosis(skew, p, T, Td, parcel) -> None:
    """
    Overlay the parcel trajectory on the Skew-T sounding plot,
    and save the figure.

    Parameters
    ----------
    skew : metpy.plots.SkewT
        Active SkewT axes wrapper returned by ``init_skewt_figure()``.
    p : pint.Quantity
        Pressure profile in hPa, shape (N,).
    T : pint.Quantity
        Environmental temperature profile in °C, shape (N,).
    Td : pint.Quantity
        Environmental dew-point profile in °C, shape (N,).
    parcel : pint.Quantity
        Parcel temperature profile in °C, shape (N,), from ``mpcalc.parcel_profile()``.
    """
    skew.plot(p, parcel, "k-", linewidth=SKEWT_PARCEL_LINEWIDTH, label="Parcel")
    skew.ax.legend(loc="upper right", fontsize=round(FONT_LABEL * FONT_SCALE_CONTOUR))


def plot_skewt_convective_levels(p, T, Td, parcel, *, lcl_p, lfc_p, el_p,
                                  p_min, p_max, t_min, t_max,
                                  station_id=None, valid_dt=None,
                                  output_dir="../outputs") -> tuple:
    """
    Overlay the LCL, LFC, and EL levels as horizontal lines with labels on the
    Skew-T sounding and parcel diagnosis plot, and save the figure. Optionally 
    annotate CAPE and CIN in the upper-left corner.

    Parameters
    ----------
    p : pint.Quantity
        Pressure profile in hPa, shape (N,).
    T : pint.Quantity
        Environmental temperature profile in °C, shape (N,).
    Td : pint.Quantity
        Environmental dew-point profile in °C, shape (N,).
    parcel : pint.Quantity
        Parcel temperature profile in °C, shape (N,), from ``mpcalc.parcel_profile``.
        lcl_p : pint.Quantity or None
        LCL pressure level in hPa, or None if not computable.
    lfc_p : pint.Quantity or None
        LFC pressure level in hPa, or None if not computable.
    el_p : pint.Quantity or None
        EL pressure level in hPa, or None if not computable.
    cape : pint.Quantity or None
        CAPE in J/kg. If provided together with cin, annotated in the upper-left corner.
    cin : pint.Quantity or None
        CIN in J/kg. If provided together with cape, annotated in the upper-left corner.
    output_dir : str or None
        Directory to save the figure. If None, the figure is not saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, metpy.plots.SkewT]
        The figure and the SkewT axes wrapper with the full parcel diagnosis overlaid.
    """
    fig, skew = plot_skewt_sounding(p, T, Td, p_min, p_max, t_min, t_max)
    if parcel is not None:
        plot_skewt_parcel_diagnosis(skew, p, T, Td, parcel)
    if station_id is not None or valid_dt is not None:
        title_y = skew.ax.get_position().y1 + 0.01
        if station_id is not None:
            fig.text(0.12, title_y, f"{station_id} Sounding",
                     fontsize=FONT_TITLE * FONT_SCALE_CONTOUR, fontweight="bold", # font size scaled down via FONT_SCALE_CONTOUR to avoid clashing with cell title
                     va="bottom", ha="left")
        if valid_dt is not None:
            fig.text(0.90, title_y, valid_dt.strftime("Valid time: %Y-%m-%d %H:%M UTC"),
                     fontsize=FONT_TITLE * FONT_SCALE_CONTOUR, # font size scaled down via FONT_SCALE_CONTOUR to avoid clashing with cell title
                     va="bottom", ha="right")
    trans = blended_transform_factory(skew.ax.transAxes, skew.ax.transData)
    _label_va = {"LCL": ("top", +8), "LFC": ("bottom", -8), "EL": ("top", +8)}
    for label, p_lev in [("LCL", lcl_p), ("LFC", lfc_p), ("EL", el_p)]:
        if p_lev is None or np.isnan(p_lev.m):
            continue
        skew.ax.plot(
            [SKEWT_LEVEL_LINE_X_START, SKEWT_LEVEL_LINE_X_END], [p_lev.m, p_lev.m],
            color=SKEWT_MARKER_COLOR, linewidth=SKEWT_LEVEL_LINE_WIDTH,
            alpha=SKEWT_LEVEL_LINE_ALPHA, transform=trans,
            solid_capstyle="butt", zorder=5,
        )
        va, dp = _label_va[label]
        skew.ax.text(
            SKEWT_LEVEL_LABEL_X, p_lev.m + dp, label,
            transform=trans, fontsize=SKEWT_LEVEL_LABEL_FONTSIZE,
            va=va, ha="left", color=SKEWT_MARKER_COLOR,
            fontweight="bold", zorder=5,
        )
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        out_path = out_dir / f"skewt_convective_levels_{timestamp}.png"
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        print(f"[OK] Saved: {out_path.resolve()}")
    return fig, skew


def plot_skewt_cape_cin(skew, p, T, parcel, *,
                        lcl_p=None, el_p=None,
                        cape=None, cin=None,
                        fig=None, output_dir="../outputs") -> None:
    """
    Shade CAPE (red) and CIN (blue) areas between the parcel path and the
    environmental temperature profile on an existing Skew-T axes.
    Optionally annotate CAPE and CIN values and save the figure.

    Parameters
    ----------
    skew : metpy.plots.SkewT
        Active SkewT axes wrapper returned by ``init_skewt_figure()``.
    p : pint.Quantity
        Pressure profile in hPa, shape (N,).
    T : pint.Quantity
        Environmental temperature profile in °C, shape (N,).
    parcel : pint.Quantity
        Parcel temperature profile in °C, shape (N,), from ``mpcalc.parcel_profile()``.
    lcl_p : pint.Quantity or None
        LCL pressure level in hPa.
    el_p : pint.Quantity or None
        EL pressure level in hPa. 
    cape : pint.Quantity or None
        CAPE in J/kg. When provided together with cin, adds annotation box.
    cin : pint.Quantity or None
        CIN in J/kg. When provided together with cape, adds annotation box.
    fig : matplotlib.figure.Figure or None
        Figure object used for saving. Required when output_dir is not None.
    output_dir : str or None
        Directory to save the figure. Skipped if None or fig is None.
    """
    parcel = parcel.copy()
    if lcl_p is not None and not np.isnan(lcl_p.m):
        below_lcl = np.asarray(p > lcl_p, dtype=bool)
        parcel[below_lcl] = T[below_lcl].to(parcel.units)
    if el_p is not None and not np.isnan(el_p.m):
        above_el = np.asarray(p < el_p, dtype=bool)
        parcel[above_el] = T[above_el].to(parcel.units)
    skew.shade_cape(p, T, parcel, alpha=SKEWT_CAPE_ALPHA)
    skew.shade_cin(p, T, parcel, alpha=SKEWT_CIN_ALPHA)
    if cape is not None and cin is not None:
        _add_skewt_annot_box(skew, [
            f"CIN  : {cin.m:.1f} J/kg",
            f"CAPE : {cape.m:.1f} J/kg",
        ])
    if fig is not None and output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        out_path = out_dir / f"skewt_cape_cin_{timestamp}.png"
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        print(f"[OK] Saved: {out_path.resolve()}")


def plot_skewt_instability_panel(p, T, Td, parcel, *, lcl_p, lfc_p, el_p,
                                  mu_cape, k,
                                  cape=None, cin=None, annotate=True,
                                  p_min, p_max, t_min, t_max,
                                  station_id=None, valid_dt=None,
                                  output_dir="../outputs") -> tuple:
    """
    Overlay the full parcel diagnosis with a single annotation box in the
    upper-left corner summarizing CAPE, CIN, MUCAPE, and K-index.

    Parameters
    ----------
    p : pint.Quantity
        Pressure profile in hPa, shape (N,).
    T : pint.Quantity
        Environmental temperature profile in °C, shape (N,).
    Td : pint.Quantity
        Environmental dew-point profile in °C, shape (N,).
    parcel : pint.Quantity
        Parcel temperature profile in °C, shape (N,), from ``mpcalc.parcel_profile()``.
    lcl_p : pint.Quantity or None
        LCL pressure level in hPa, or None if not computable.
    lfc_p : pint.Quantity or None
        LFC pressure level in hPa, or None if not computable.
    el_p : pint.Quantity or None
        EL pressure level in hPa, or None if not computable.
    mu_cape : pint.Quantity
        Most Unstable Convective Available Potential Energy in J/kg.
    k : pint.Quantity
        K-index in °C.
    cape : pint.Quantity or None
        Surface-based CAPE in J/kg. When provided together with cin,
        prepended to the annotation box.
    cin : pint.Quantity or None
        Surface-based CIN in J/kg. When provided together with cape,
        prepended to the annotation box.
    annotate : bool
        If False, skip drawing the annotation box (used internally by
        ``plot_skewt_winds`` to avoid duplicate boxes).
    output_dir : str or None
        Directory to save the figure. If None, the figure is not saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, metpy.plots.SkewT]
        The figure and the SkewT axes wrapper with the full parcel diagnosis and
        the instability annotation overlaid.
    """
    fig, skew = plot_skewt_convective_levels(
        p, T, Td, parcel, lcl_p=lcl_p, lfc_p=lfc_p, el_p=el_p,
        p_min=p_min, p_max=p_max, t_min=t_min, t_max=t_max,
        output_dir=None,
    )
    if station_id is not None or valid_dt is not None:
        title_y = skew.ax.get_position().y1 + 0.01
        if station_id is not None:
            fig.text(0.12, title_y, f"{station_id} Sounding",
                     fontsize=FONT_TITLE * FONT_SCALE_CONTOUR, fontweight="bold", # font size scaled down via FONT_SCALE_CONTOUR to avoid clashing with cell title
                     va="bottom", ha="left")
        if valid_dt is not None:
            fig.text(0.90, title_y, valid_dt.strftime("Valid time: %Y-%m-%d %H:%M UTC"),
                     fontsize=FONT_TITLE * FONT_SCALE_CONTOUR, # font size scaled down via FONT_SCALE_CONTOUR to avoid clashing with cell title
                     va="bottom", ha="right")
    if annotate:
        lines = []
        if cape is not None and cin is not None:
            lines.append(f"CIN     : {cin.m:.1f} J/kg")
            lines.append(f"CAPE    : {cape.m:.1f} J/kg")
        lines.append(f"MUCAPE  : {mu_cape.m:.1f} J/kg")
        lines.append(f"K-index : {k.m:.1f} °C")
        _add_skewt_annot_box(skew, lines)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        out_path = out_dir / f"skewt_instability_panel_{timestamp}.png"
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        print(f"[OK] Saved: {out_path.resolve()}")
    return fig, skew


def plot_skewt_winds(p, T, Td, parcel, u, v, *, lcl_p, lfc_p, el_p,
                     mu_cape, k, shear_mag,
                     cape=None, cin=None,
                     p_min, p_max, t_min, t_max,
                     station_id=None, valid_dt=None,
                     output_dir="../outputs") -> tuple:
    """
    Overlay wind barbs along the right margin of the Skew-T diagram with the
    full parcel diagnosis and a single upper-left annotation box summarizing
    CAPE, CIN, MUCAPE, K-index, and 0–6 km bulk shear.

    Parameters
    ----------
    p : pint.Quantity
        Pressure profile in hPa, shape (N,).
    T : pint.Quantity
        Environmental temperature profile in °C, shape (N,).
    Td : pint.Quantity
        Environmental dew-point profile in °C, shape (N,).
    parcel : pint.Quantity
        Parcel temperature profile in °C, shape (N,), from ``mpcalc.parcel_profile``.
    u : pint.Quantity
        Zonal wind component in knots, shape (N,).
    v : pint.Quantity
        Meridional wind component in knots, shape (N,).
    lcl_p : pint.Quantity or None
        LCL pressure level in hPa, or None if not computable.
    lfc_p : pint.Quantity or None
        LFC pressure level in hPa, or None if not computable.
    el_p : pint.Quantity or None
        EL pressure level in hPa, or None if not computable.
    mu_cape : pint.Quantity
        Most Unstable Convective Available Potential Energy in J/kg.
    k : pint.Quantity
        K-index in °C.
    shear_mag : float
        0–6 km bulk shear magnitude in knots.
    cape : pint.Quantity or None
        Surface-based CAPE in J/kg. When provided together with cin,
        prepended to the annotation box.
    cin : pint.Quantity or None
        Surface-based CIN in J/kg. When provided together with cape,
        prepended to the annotation box.
    output_dir : str or None
        Directory to save the figure. If None, the figure is not saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, metpy.plots.SkewT]
        The figure and the SkewT axes wrapper with the full parcel diagnosis,
        instability annotation, and wind barbs overlaid.
    """
    fig, skew = plot_skewt_instability_panel(
        p, T, Td, parcel, lcl_p=lcl_p, lfc_p=lfc_p, el_p=el_p,
        mu_cape=mu_cape, k=k,
        p_min=p_min, p_max=p_max, t_min=t_min, t_max=t_max,
        annotate=False, output_dir=None,
    )
    if station_id is not None or valid_dt is not None:
        title_y = skew.ax.get_position().y1 + 0.01
        if station_id is not None:
            fig.text(0.12, title_y, f"{station_id} Sounding",
                     fontsize=FONT_TITLE * FONT_SCALE_CONTOUR, fontweight="bold", # font size is scaled down based on font_scale_contour to avoid duplicates
                     va="bottom", ha="left")
        if valid_dt is not None:
            fig.text(0.90, title_y, valid_dt.strftime("Valid time: %Y-%m-%d %H:%M UTC"),
                     fontsize=FONT_TITLE * FONT_SCALE_CONTOUR, # font size is scaled down based on font_scale_contour to avoid duplicates
                     va="bottom", ha="right")
    mask = p.m >= 100
    skew.plot_barbs(p[mask][::SKEWT_WIND_STRIDE], u[mask][::SKEWT_WIND_STRIDE],
                    v[mask][::SKEWT_WIND_STRIDE], length=BARB_LENGTH * BARB_SCALE_UPPER)
    lines = []
    if cape is not None and cin is not None:
        lines.append(f"CIN     : {cin.m:.1f} J/kg")
        lines.append(f"CAPE    : {cape.m:.1f} J/kg")
    lines.append(f"MUCAPE  : {mu_cape.m:.1f} J/kg")
    lines.append(f"K-index : {k.m:.1f} °C")
    lines.append(f"0–6 km shear : {shear_mag:.1f} kt")
    _add_skewt_annot_box(skew, lines)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        out_path = out_dir / f"skewt_wind_profile_{timestamp}.png"
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        print(f"[OK] Saved: {out_path.resolve()}")
    return fig, skew
