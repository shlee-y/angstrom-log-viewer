# Angstrom Log Viewer

A web-based viewer for analyzing Angstrom Engineering deposition tool logs. Supports multiple machines in one unified interface.

Currently supported:
- **JJ Evaporator** — Al/AlOx/Al Josephson junction e-beam evaporator
- **Ta Sputterer** — Tantalum pulsed DC sputtering for superconducting resonators

## Quick start

### 1. Install Python dependencies

Python 3.10+ with Flask ≥ 2.2, pandas, and numpy:

```bash
pip install flask>=2.2 pandas numpy
```

### 2. Run the viewer

**Unified viewer** (both machines, recommended):

```bash
python app.py --jj-dir "path/to/jj/logs" --ta-dir "path/to/ta/logs"
```

Then open **http://localhost:5000** in your browser. Toggle between machines using the segmented control in the header.

You can omit either `--jj-dir` or `--ta-dir` if you only have one machine's logs.

**Single-machine viewers** (still work independently):

```bash
python server.py --log-dir "path/to/jj/logs"           # JJ only, port 5000
python ta-server.py --log-dir "path/to/ta/logs" --port 5001  # Ta only, port 5001
```

To use a different port: add `--port 8080`.

### 3. Expected log folder structure

**JJ Evaporator** — one subfolder per run:

```
<jj-log-dir>/
  CQtm_<RecipeName>-<user>_YYYYMMDD_HHMMSS/
    *.csv, *_details.json, *_Original_*.xml, *_Complete_*.xml, *_Status_*.xml
```

**Ta Sputterer** — flat files (no subfolders):

```
<ta-log-dir>/
  SHatlab_*_YYYYMMDD_HHMMSS.csv
  SHatlab_*_YYYYMMDD_HHMMSS_details.json
  <RecipeName>_Original_*.xml, *_Complete_*.xml, *_Status_*.xml
```

The viewer auto-detects all valid runs in each directory. You can also change directories at runtime via the gear icon (⚙) in the header.

## Features

### Summary tab
- Layer-by-layer (JJ) or phase-by-phase (Ta) cards with key metrics
- Color-coded target deltas (green/amber/red)
- Substrate tilt/rotation angles per step
- Sensor calibration info via "CAL" chip on each card
- DC vs Pulsed DC mode indicator (Ta)
- Click any card to jump to its time range in Time Series

### Time Series tab
- Stacked synchronized chart panels (rate, power, thickness, pressure, temperature)
- Shutter/valve timing diagram lanes below the charts
- Phase boundary markers with labels
- **Interactions:**
  - Drag left-right → x-axis zoom
  - Scroll wheel → y-axis zoom (per panel)
  - Right-click drag → y-axis pan
  - Hover → crosshair + tooltip with all signal values
  - Double-click → reset all zoom

### Comparison tab
- Dropdown multi-select to pick which runs to compare (loads on demand, scales to 100+ runs)
- Metrics table with sparkline trends
- Anomaly highlighting (>1σ amber, >2σ red from median)
- Collapsible metric groups

### Recipe Diff tab
- Compares Original (as-loaded) vs Complete (as-run) recipe XML
- Shows mid-run parameter edits with old → new values
- "Show all parameters" toggle

### Valves tab
- Full-size digital timing diagram for all shutters/valves
- Same zoom and crosshair interactions as Time Series

## Files

| File | Description |
|------|-------------|
| `app.py` | **Unified backend** — serves both machines on one port |
| `app.html` | **Unified frontend** — machine toggle, all features |
| `server.py` | JJ evaporator standalone backend |
| `index.html` | JJ evaporator standalone frontend |
| `ta-server.py` | Ta sputterer standalone backend |
| `ta-index.html` | Ta sputterer standalone frontend |

## Notes

- Pressure values of `-9999` in the CSV are sentinel values ("no valid reading") and are automatically masked
- Truncated/incomplete CSVs are handled gracefully and flagged in the run selector
- Parsed run data is cached in memory after first load for fast access
- The Ta sputterer uses calibrated deposition (QCM calibrates rate, then deposits by power control) — the "Film thickness" trace shows the calibrated estimate, which is the actual film thickness
