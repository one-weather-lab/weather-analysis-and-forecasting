#!/usr/bin/env python3
"""
Script Name: plot_helpers.py
Purpose: Surface and upper-air map visualizations for METAR and radiosonde networks.
         Isobars, isotherms, station models, and 500 hPa station plots.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-04-13
Version: 2.0.0
License: MIT

Notes:
  • Inputs:  (1) Decoded, QC-passed METAR DataFrame (station, valid, temp_c, relh,
                 wspd, wdir, u_kt, v_kt) + OurAirports coordinate table for surface maps.
             (2) Pre-gridded fields (grid_lon, grid_lat, mslp_grid, temp_grid,
                 z500_grid) produced by contouring_helpers.py for contour-based
                 isobar, isotherm, and upper-air map functions.
             (3) RAOB DataFrame from fetch_europe_raob_fields() (raob_helpers.py)
                 for the 500 hPa station plot.
  • Outputs: pd.DataFrame (one representative row per station, merged with
             coordinates); PNG figures saved to the specified output directory.
  • Configuration: All tunable values are module-level constants: figure size,
                   DPI, wind-barb length, label offsets, font sizes and scale
                   factors (FONT_SCALE_*, BARB_SCALE_*, LABEL_SCALE_*), thinning
                   radii, Lambert Conformal projection parameters
                   (PROJ_CENTRAL_LON/LAT, PROJ_STD_PARALLELS), and pressure
                   center detection parameters (N_SIZE, HL_MIN_SEP_DEG,
                   ISOTHERM_MIN_AREA_KM2).
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import logging

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, minimum_filter
from metpy.calc import reduce_point_density

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

# [Barb scale factors]
BARB_SCALE_STATION  = 0.80   # barb length scale for continental network plot

# [Label scale factors — Europe map covers ~4× the lat/lon range of Greece]
LABEL_SCALE_STATION = 3.0    # multiplier for lon/lat label offsets on continental network plot

# [Projection — Lambert Conformal for the European domain]
PROJ_CENTRAL_LON    = 15          # central longitude (degrees East)
PROJ_CENTRAL_LAT    = 50          # central latitude  (degrees North)
PROJ_STD_PARALLELS  = (35, 65)    # standard parallels

# [Pressure center detection]
N_SIZE          = 25         # neighbourhood size for extremum filter
SYMBOL_SIZE     = 20         # font size for H/L symbol text
HL_MIN_SEP_DEG  = 25.0       # minimum separation between H/L centers (degrees)
ISOTHERM_MIN_AREA_KM2 = 200000  # suppress closed isotherms enclosing less than this area (km²)

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
            is_closed = np.allclose(seg[0], seg[-1], atol=0.05)
            if not is_closed:
                kept_segs.append(seg)
                continue
            # Area via shoelace on km coordinates
            x, y = seg[:, 0], seg[:, 1]
            mean_lat = float(np.mean(y))
            kx = x * 111.32 * np.cos(np.radians(mean_lat))
            ky = y * 111.32
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
        if not (lon_min_e <= lon_val <= lon_max_e and
                lat_min_e <= lat_val <= lat_max_e):
            continue

        val = data[iy, ix]

        ax.text(
            lon_val, lat_val, label,
            color=color, fontsize=symbol_size, fontweight="bold",
            ha="center", va="center",
            transform=transform, zorder=10,
        )
        ax.text(
            lon_val, lat_val - 0.8, f"{val:.0f}",
            color=color, fontsize=symbol_size - 2,
            ha="center", va="top",
            transform=transform, zorder=10,
        )

# [Output / Export]

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
            s=np.clip(df_vrb["wspd"] * 1.6, VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
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

    ax.set_title(title_line, fontsize=FONT_TITLE, loc="left")
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


def _setup_europe_map(ax=None):
    """
    Create or configure a Lambert Conformal Europe-domain map.

    If *ax* is None a new figure and axes are created; otherwise the
    existing axes are configured in place.

    Returns
    -------
    tuple (fig, ax, proj, data_crs)
    """
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

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray", alpha=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

    return fig, ax, proj, data_crs


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
        t_max = obs_times.max()
        time_label = t_max.strftime("%Y-%m-%d %H:%M UTC")
    else:
        time_label = "unknown time"

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
        time_label = obs_times.max().strftime("%Y-%m-%d %H:%M UTC") if len(obs_times) > 0 else "unknown time"
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
        time_label = obs_times.max().strftime("%Y-%m-%d %H:%M UTC") if len(obs_times) > 0 else "unknown time"
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
        Must contain ``longitude``, ``latitude``, ``valid``, ``temp_c``,
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
    time_label = obs_times.max().strftime("%Y-%m-%d %H:%M UTC") if len(obs_times) > 0 else "unknown time"

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
            s=np.clip(df_thin.loc[is_vrb, "wspd"] * 1.4, VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
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
        Must contain ``longitude``, ``latitude``, ``valid``, ``wdir``,
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
    time_label = obs_times.max().strftime("%Y-%m-%d %H:%M UTC") if len(obs_times) > 0 else "unknown time"

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
            s=np.clip(df_thin.loc[is_vrb, "wspd"] * 1.6, VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
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
        Must contain ``longitude``, ``latitude``, ``valid``, ``wdir``,
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
    time_label = obs_times.max().strftime("%Y-%m-%d %H:%M UTC") if len(obs_times) > 0 else "unknown time"

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
        gridspec_kw={"wspace": 0.04},
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
            s=np.clip(df_thin.loc[is_vrb, "wspd"] * 1.6, VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
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
    isotherm_levels = np.arange(-40, 45, isotherm_interval)
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

    * **Left** — MSLP isobars, H/L pressure centers, thinned wind barbs, and
      thinned station relative humidity (%) in blue at the lower-left of each
      station position, matching the layout of ``plot_europe_metar_network``.
    * **Right** — temperature isotherms only (red).

    The observation time is derived automatically from the ``valid`` column
    of *df_plot* and included in each panel title.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Plot-ready observation DataFrame from ``build_europe_network_plot_df()``.
        Must contain ``longitude``, ``latitude``, ``valid``, ``wdir``,
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
    time_label = obs_times.max().strftime("%Y-%m-%d %H:%M UTC") if len(obs_times) > 0 else "unknown time"

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
        gridspec_kw={"wspace": 0.04},
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
            s=np.clip(df_thin.loc[is_vrb, "wspd"] * 1.6, VRB_SCATTER_MIN_S, VRB_SCATTER_MAX_S),
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
    isotherm_levels = np.arange(-40, 45, isotherm_interval)
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

# [Upper-air station plot]

def plot_europe_500hpa_stations(
    df_raob,
    output_dir: str = "../outputs",
) -> str:
    """
    Upper-air station plot at 500 hPa for the European radiosonde network.

    Each station is drawn with:
    - Upper-left  — temperature (°C, red).
    - Upper-right — geopotential height in decameters (dam), coded as the
                    last 3 digits (e.g. 576 for 5760 m).
    - Lower-left  — dew-point depression (T - Td, °C), rounded to 1 °C.
    - Wind barbs  — 500 hPa wind in knots.

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
    _lu = 0.40
    _ll = 0.40

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
            length=BARB_LENGTH * BARB_SCALE_STATION * 0.80,
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
        f"500 hPa Station Plot \n{time_label}",
        fontsize=FONT_TITLE, loc="left",
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
