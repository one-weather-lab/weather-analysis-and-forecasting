#!/usr/bin/env python3
"""
Script Name: plot_helpers.py
Purpose: Network-scale surface weather map based on decoded METARs.

Author(s): Christos Giannaros, One Weather Lab, University of Ioannina <chris.giannaros@uoi.gr>
Last updated: 2026-02-28
Version: 1.0.0
License: MIT

Notes:
  • Inputs:  Decoded, QC-passed METAR DataFrame (station, valid, temp_c, relh, wspd,
             wdir, u_kt, v_kt) and a station coordinate table (station,
             lat, lon) from the OurAirports open database.
  • Outputs: pd.DataFrame (one representative row per station, merged with
             coordinates); PNG figure saved to the specified output directory.
  • Configuration: All tunable values (DPI, figure size, wind-barb length, label
                   offsets, font sizes) are defined as module-level constants.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# [Data sources]
AIRPORTS_URL = (
    "https://raw.githubusercontent.com/davidmegginson/"
    "ourairports-data/master/airports.csv"
)
ISO_COUNTRY = "GR"        # Two-letter ISO country code for network filtering

# [Map geometry]
GREECE_EXTENT  = [19.0, 29.5, 34.5, 42.0]   # [lon_min, lon_max, lat_min, lat_max]

# [Figure output]
FIG_SIZE_IN    = (20, 20)    # figure size in inches
FIG_DPI        = 300         # output DPI for saved PNG

# [Wind rendering]
CALM_THRESHOLD_KT = 1        # speed (kt) below which wind is considered calm
BARB_LENGTH        = 10      # barb length in points
VRB_SCATTER_MIN_S  = 8       # minimum scatter marker size for VRB wind
VRB_SCATTER_MAX_S  = 60      # maximum scatter marker size for VRB wind
CALM_CIRCLE_S      = 36     # scatter marker size for calm wind circles

# [Label layout]
LABEL_LON_OFFSET = 0.12     # longitude offset for T / RH text labels (degrees)
LABEL_LAT_UPPER  = 0.07     # latitude offset upward for temperature label
LABEL_LAT_LOWER  = 0.07     # latitude offset downward for RH label

# [Font sizes]
FONT_LABEL   = 15            # temperature and RH text
FONT_COUNT   = 18            # station count annotation
FONT_TITLE   = 22            # map title

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

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

# [Output / Export]

def plot_greece_metar_network(
    df_plot: pd.DataFrame,
    output_dir: str = "../outputs",
    title_suffix: Optional[str] = None,
) -> str:
    """
    Draw simplified station plots for the METAR network.

    For each station the following elements are drawn:

    - **Wind barbs** (knots, black) for stations with a known direction
    - **VRB scatter** (black dot, size proportional to speed) for variable-
      direction wind
    - **Calm circle** (open black circle) for wind speed <= CALM_THRESHOLD_KT kt
    - **Temperature** (degC, red text, upper-right of position)
    - **Relative Humidity** (%, blue text, lower-right of position)

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
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

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

    # ── Labels: Temperature (red) + RH (blue) ───────────────────────────
    for _, rr in df.iterrows():
        if pd.notna(rr.get("temp_c", np.nan)):
            ax.text(
                rr["lon"] + LABEL_LON_OFFSET, rr["lat"] + LABEL_LAT_UPPER,
                f"{rr['temp_c']:.0f} \u00b0C",
                color="red", fontsize=FONT_LABEL,
                transform=proj, clip_on=True,
            )
        if pd.notna(rr.get("relh", np.nan)):
            ax.text(
                rr["lon"] + LABEL_LON_OFFSET, rr["lat"] - LABEL_LAT_LOWER,
                f"{rr['relh']:.0f}%",
                color="blue", fontsize=FONT_LABEL,
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
