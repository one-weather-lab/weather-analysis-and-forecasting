# 🦉 Weather Analysis and Forecasting

## Notebooks

| # | Title | Core topics |
|---|-------|-------------|
| 01 | surface_weather_reports | Application of a surface **weather data analysis workflow** that reflects **operational meteorology** practices |
| 02 | synoptic_analysis_I | Application of a **synoptic analysis** workflow through surface contouring, and **upper-air station plotting** at 500 hPa |


> Notebooks are added progressively throughout the course.

## Running the Notebooks

Two execution paths are available.

### Option A — Binder (no installation)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/one-weather-lab/weather-analysis-and-forecasting/HEAD)

Open any notebook and run cells sequentially.
The environment builds automatically. No local setup is required.

### Option B — Local machine

Clone the repository:
```bash

git clone https://github.com/one-weather-lab/weather-analysis-and-forecasting.git
cd weather-analysis-and-forecasting
```

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate weather-analysis-and-forecasting
```

Launch Jupyter Lab:

```bash
jupyter lab notebooks/
```

If you prefer pip over conda: `pip install -r requirements.txt` covers the core
dependencies.