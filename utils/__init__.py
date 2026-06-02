# utils package

# -----------------------------------------------------------------------------
# Data fetching — surface
# -----------------------------------------------------------------------------
from .iem_raw import fetch_iem_raw
from .noaa_realtime import fetch_noaa_realtime

# -----------------------------------------------------------------------------
# Parsing and QC
# -----------------------------------------------------------------------------
from .metar_helpers import (
    # METAR string parser
    parse_metar_string,
    decode_metar_row,
    # Unit conversion helpers
    knots_to_ms,
    knots_to_kmh,
    knots_to_beaufort,
    knots_to_uv_barbs,
    inhg_to_hpa,
    sky_cover_from_code,
    # QC check functions
    check_dewpoint_consistency,
    check_rh_range,
    check_wind_temporal_spike,
)

# -----------------------------------------------------------------------------
# Objective analysis / gridding
# -----------------------------------------------------------------------------
from .contouring_helpers import (
    build_europe_grid,
    grid_variable,
    smooth_grid,
    grid_surface_fields,
)

# -----------------------------------------------------------------------------
# Data fetching — GFS analysis and forecast
# -----------------------------------------------------------------------------
from .herbie_gfs import (
    resolve_gfs_data_folder,
    fetch_gfs_analysis,
    fetch_gfs_forecast,
    build_gfs_forecast_ds,
)

# -----------------------------------------------------------------------------
# Data fetching — GEFS ensemble
# -----------------------------------------------------------------------------
from .herbie_gefs import resolve_gefs_data, fetch_gefs_members, build_gefs_forecast_da

# -----------------------------------------------------------------------------
# Nearest-grid-point lookup and ensemble meteogram
# -----------------------------------------------------------------------------
from .meteogram_helpers import get_nearest_grid_point, build_meteogram

# -----------------------------------------------------------------------------
# Diagnostics — GFS analysis
# -----------------------------------------------------------------------------
from .gfs_diagnostics import (
    compute_temperature_advection,
    compute_relative_vorticity,
    compute_wind_speed,
)

# -----------------------------------------------------------------------------
# Data fetching — upper air
# -----------------------------------------------------------------------------
from .wyoming_raob import (
    fetch_latest_sounding,
    fetch_retrospective_sounding,
)
from .raob_helpers import (
    fetch_igra2_europe,
    fetch_europe_raob_fields,
)

# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------
from .plot_helpers import (
    # Station coordinate helpers
    fetch_station_coords,
    fetch_ourairports_europe,
    # Plot-ready DataFrame builders
    build_network_plot_df,
    build_europe_network_plot_df,
    # Country-scale maps
    plot_greece_metar_network,
    # Europe-scale surface maps
    plot_europe_metar_network,
    plot_europe_mslp_raw,
    plot_europe_isobars,
    plot_maxmin_points,
    plot_europe_isobars_hl,
    plot_europe_enhanced_station_isobars,
    plot_europe_isobars_wind,
    plot_europe_isobars_temperature,
    plot_europe_isobars_temperature_humidity,
    # Upper-air station plot
    plot_europe_500hpa_stations,
    # Upper-air maps (GFS analysis)
    plot_850hpa_gph_temperature_wind,
    plot_850hpa_temperature_advection,
    plot_700hpa_relative_humidity,
    plot_500hpa_gph,
    plot_500hpa_relative_vorticity,
    plot_500hpa_relative_vorticity_advection,
    plot_250hpa_jet,
    plot_gfs_surface_chart,
    plot_ensemble_mean_spread,
    plot_compound_probability_animation,
    plot_upper_air_overview,
    plot_four_panel_forecast_animation,
    # Skew-T diagrams
    init_skewt_figure,
    add_skewt_mixing_lines,
    add_skewt_dry_adiabats,
    add_skewt_moist_adiabats,
    plot_skewt_sounding,
    plot_skewt_parcel_diagnosis,
    plot_skewt_convective_levels,
    plot_skewt_instability_panel,
    plot_skewt_winds,
)

__all__ = [
    # iem_raw
    "fetch_iem_raw",
    # noaa_realtime
    "fetch_noaa_realtime",
    # metar_helpers
    "parse_metar_string",
    "decode_metar_row",
    "knots_to_ms",
    "knots_to_kmh",
    "knots_to_beaufort",
    "knots_to_uv_barbs",
    "inhg_to_hpa",
    "sky_cover_from_code",
    "check_dewpoint_consistency",
    "check_rh_range",
    "check_wind_temporal_spike",
    # contouring_helpers
    "build_europe_grid",
    "grid_variable",
    "smooth_grid",
    "grid_surface_fields",
    # wyoming_raob
    "fetch_latest_sounding",
    "fetch_retrospective_sounding",
    # raob_helpers
    "fetch_igra2_europe",
    "fetch_europe_raob_fields",
    # plot_helpers — coordinate helpers
    "fetch_station_coords",
    "fetch_ourairports_europe",
    # plot_helpers — DataFrame builders
    "build_network_plot_df",
    "build_europe_network_plot_df",
    # plot_helpers — country-scale maps
    "plot_greece_metar_network",
    # plot_helpers — Europe-scale surface maps
    "plot_europe_metar_network",
    "plot_europe_mslp_raw",
    "plot_europe_isobars",
    "plot_europe_isobars_hl",
    "plot_europe_enhanced_station_isobars",
    "plot_europe_isobars_wind",
    "plot_europe_isobars_temperature",
    "plot_europe_isobars_temperature_humidity",
    "plot_maxmin_points",
    # plot_helpers — upper-air station plot
    "plot_europe_500hpa_stations",
    # herbie_gfs
    "resolve_gfs_data_folder",
    "fetch_gfs_analysis",
    "fetch_gfs_forecast",
    "build_gfs_forecast_ds",
    # herbie_gefs
    "resolve_gefs_data",
    "fetch_gefs_members",
    "build_gefs_forecast_da",
    # meteogram_helpers
    "get_nearest_grid_point",
    "build_meteogram",
    # gfs_diagnostics
    "compute_temperature_advection",
    "compute_relative_vorticity",
    "compute_wind_speed",
    # plot_helpers — upper-air maps (GFS analysis)
    "plot_850hpa_gph_temperature_wind",
    "plot_850hpa_temperature_advection",
    "plot_700hpa_relative_humidity",
    "plot_500hpa_gph",
    "plot_500hpa_relative_vorticity",
    "plot_500hpa_relative_vorticity_advection",
    "plot_250hpa_jet",
    "plot_gfs_surface_chart",
    "plot_ensemble_mean_spread",
    "plot_compound_probability_animation",
    "plot_upper_air_overview",
    "plot_four_panel_forecast_animation",
    # plot_helpers — skew-T diagrams
    "init_skewt_figure",
    "add_skewt_mixing_lines",
    "add_skewt_dry_adiabats",
    "add_skewt_moist_adiabats",
    "plot_skewt_sounding",
    "plot_skewt_parcel_diagnosis",
    "plot_skewt_convective_levels",
    "plot_skewt_instability_panel",
    "plot_skewt_winds",
]
