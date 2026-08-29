"""
Flask backend for Angstrom Ta sputterer log viewer.
Serves CSV time-series data, phase summaries, recipe diffs,
and cross-run comparisons for Angstrom EvoVac Ta sputter runs.
"""

import os
import re
import json
import math
import glob as globmod

from flask import Flask, jsonify, request, send_from_directory, abort
from flask.json.provider import DefaultJSONProvider
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, 'Ta log')

# CSV files are flat: *_YYYYMMDD_HHMMSS.csv
CSV_RE = re.compile(r'_(\d{8}_\d{6})\.csv$')


class NumpyJSONProvider(DefaultJSONProvider):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

# Parsed run data cache:  run_id -> dict of numpy arrays
_cache = {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default phase names; populated from recipe XML when available
PHASE_NAMES = {
    1: 'Temperature control 450',
    2: 'Temperature control 800',
    3: 'Ignite Plasma & Ramp Power',
    4: 'Calibrate Rate',
    5: 'Deposit Calibrated Rate',
    6: 'PostCondition1',
    7: 'Post Condition 2',
}

# XML namespaces used in the Angstrom recipe files
NS = {
    'r': 'http://schemas.datacontract.org/2004/07/AE.DepControl.DepositionRecipe',
    'a': 'http://schemas.datacontract.org/2004/07/AE.DepControl.DepositionRecipe.Actions',
    'i': 'http://www.w3.org/2001/XMLSchema-instance',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_elapsed(s):
    """Parse 'H:MM:SS.fffffff' elapsed-time string to total seconds."""
    s = str(s).strip()
    m = re.match(r'(\d+):(\d{2}):(\d{2})(?:\.(\d+))?', s)
    if not m:
        return 0.0
    h, mi, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
    frac = float('0.' + m.group(4)) if m.group(4) else 0.0
    return h * 3600 + mi * 60 + sec + frac


def find_file(directory, pattern):
    """Return first file matching *pattern* inside *directory*, or None."""
    matches = globmod.glob(os.path.join(directory, pattern))
    return matches[0] if matches else None


def read_last_csv_line(path):
    """Read the last non-empty line from a CSV without loading the whole file."""
    with open(path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 8000))
        data = f.read().decode('utf-8', errors='replace')
    lines = [l for l in data.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ''


def clean(arr):
    """Convert numpy array to a JSON-safe Python list (NaN/Inf -> None)."""
    result = arr.tolist()
    if arr.dtype.kind == 'f':
        return [None if (v != v or v == float('inf') or v == float('-inf'))
                else v for v in result]
    return result


# ---------------------------------------------------------------------------
# Column mapping via _details.json
# ---------------------------------------------------------------------------

def build_col_map(details_path):
    """Build logical-name -> CSV-column-index map.

    The _details.json entries correspond to CSV columns 5+
    (columns 0-4 are: Version, Date, Time, Elapsed Time, Triggered).
    We use the *Name* field, keeping the first occurrence for names
    that repeat across source blocks.
    """
    with open(details_path, encoding='utf-8') as f:
        details = json.load(f)

    n2c = {}
    for i, entry in enumerate(details):
        name = entry['Name']
        if name not in n2c:
            n2c[name] = i + 5          # offset for the 5 logger columns

    m = {
        'layer':              n2c['DepSequencer.CurrentLayer'],
        'phase':              n2c['DepSequencer.CurrentPhase'],
        'step':               n2c['DepSequencer.StepName'],
        'seq_running':        n2c['DepSequencer.SequenceRunning'],
        'pressure':           n2c['Chamber[0].Pressure'],
        'process_pressure':   n2c['Chamber[0].ProcessPressure'],
        'ar_flow':            n2c['MFC[1].ActualFlow'],
        'src_rate':           n2c['Source[11].ActualRate'],
        'src_thickness':      n2c['Source[11].ActualThickness'],
        'src_power_pct':      n2c['Source[11].ActualPower'],
        'src_power_setpoint': n2c['Source[11].PowerSetpoint'],
        'src_shutter':        n2c['Source[11].OpenShutter'],
        'src_ref_rate':       n2c['Source[11].ReferenceSensorRate'],
        'src_ref_thickness':  n2c['Source[11].ReferenceSensorThickness'],
        'dc_power':           n2c['PulsedDCSupply[11].ActualPower'],
        'dc_voltage':         n2c['PulsedDCSupply[11].ActualVoltage'],
        'dc_current':         n2c['PulsedDCSupply[11].ActualCurrent'],
        'spark_count':        n2c['PulsedDCSupply[11].SparkCounter'],
        'pulse_freq':         n2c['PulsedDCSupply[11].PulseFrequency'],
        'pulse_duration':     n2c['PulsedDCSupply[11].PulseDuration'],
        'dc_counter':         n2c['PulsedDCSupply[11].DCCounter'],
        'substrate_shutter':  n2c['SeqSubstrate[0].ShutterOpen'],
        'substrate_cmd':      n2c['SeqSubstrate[0].bOpenShutter'],
        'heater_temp':        n2c['Heater[1].ActualTemperature'],
        'heater_power':       n2c['Heater[1].ActualPower'],
        'heater_ref_tc':      n2c['Heater[1].ReferenceTC'],
        'heater_cal_tc':      n2c['Heater[1].CalibrationTC'],
        'pyro_temp':          n2c['kSA-ICEMetrology[0].EcpTemperature'],
        'pyro_uncorr':        n2c['kSA-ICEMetrology[0].EcpUncorrectedTemp'],
        'reflectivity':       n2c['kSA-ICEMetrology[0].EcpReflectivity'],
        'sensor_rate':        n2c['Sensor[1].rRate'],
        'sensor_thickness':   n2c['Sensor[1].rThickness'],
        'sensor_shutter':     n2c['Sensor[1].bOpenShutter'],
        'tooling':            n2c['PhysicalSensor[1].ToolingFactor'],
        'density':            n2c['PhysicalSensor[1].Density'],
        'zfactor':            n2c['PhysicalSensor[1].ZFactor'],
        'crystal_remaining':  n2c['PhysicalSensor[1].rPercentRemaining'],
        'rotation':           n2c['Servo[1].stActual.Position'],
        'z_position':         n2c['Servo[2].stActual.Position'],
        'cal_rate':           n2c['Source[11].CalibratedSource.Rate'],
        'cal_power':          n2c['Source[11].CalibratedSource.Power'],
        'cal_est_thickness':  n2c['Source[11].CalibratedSource.EstimatedThickness'],
        'cal_elapsed':        n2c['Source[11].CalibratedSource.ElapsedDepositTime'],
    }
    return m


# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------

def get_run_files(run_id):
    """Return dict of file paths for a given run_id (timestamp string)."""
    csv_path = None
    det_path = None
    orig_xml = None
    comp_xml = None

    for name in os.listdir(LOG_DIR):
        full = os.path.join(LOG_DIR, name)
        if not os.path.isfile(full):
            continue

        # Match CSV by run_id timestamp
        if name.endswith(f'{run_id}.csv'):
            csv_path = full
        elif name.endswith(f'{run_id}_details.json'):
            det_path = full
        elif '_Original_' in name and name.endswith('.xml') and run_id in name:
            # Skip Nebula cluster tool XMLs (Hatlab_* without recipe name prefix)
            if not name.startswith('Hatlab_'):
                orig_xml = full
        elif '_Complete_' in name and name.endswith('.xml'):
            if not name.startswith('Hatlab_'):
                comp_xml = full

    return {
        'csv': csv_path,
        'details': det_path,
        'original_xml': orig_xml,
        'complete_xml': comp_xml,
    }


def load_run(run_id):
    """Load a run's CSV data, parse columns, and cache the result.

    Returns a dict of numpy arrays keyed by channel name, or None on error.
    """
    if run_id in _cache:
        return _cache[run_id]

    files = get_run_files(run_id)
    if not files['csv'] or not files['details']:
        return None

    try:
        col = build_col_map(files['details'])
        use_cols = sorted(set([3] + list(col.values())))

        df = pd.read_csv(files['csv'], header=None, skiprows=1,
                         usecols=use_cols, low_memory=False)

        t = df[3].apply(parse_elapsed).values

        layer = pd.to_numeric(df[col['layer']], errors='coerce') \
                  .fillna(0).astype(int).values
        phase = pd.to_numeric(df[col['phase']], errors='coerce') \
                  .fillna(0).astype(int).values

        def fcol(key):
            return pd.to_numeric(df[col[key]], errors='coerce') \
                     .fillna(0.0).values.copy()

        def pcol(key):
            """Float column with -9999 sentinel masking."""
            v = pd.to_numeric(df[col[key]], errors='coerce').values.copy()
            v[v <= -9000] = np.nan
            return v

        def bcol(key):
            return (df[col[key]].astype(str).str.strip() == 'True').values

        data = {
            'files':            files,
            't':                t,
            'layer':            layer,
            'phase':            phase,
            'pressure':         pcol('pressure'),
            'process_pressure': fcol('process_pressure'),
            'ar_flow':          fcol('ar_flow'),
            'src_rate':         fcol('src_rate'),
            'src_thickness':    fcol('src_thickness'),
            'src_power_pct':    fcol('src_power_pct'),
            'src_shutter':      bcol('src_shutter'),
            'dc_power':         fcol('dc_power'),
            'dc_voltage':       fcol('dc_voltage'),
            'dc_current':       fcol('dc_current'),
            'spark_count':      fcol('spark_count'),
            'pulse_freq':       fcol('pulse_freq'),
            'pulse_duration':   fcol('pulse_duration'),
            'substrate_shutter': bcol('substrate_shutter'),
            'substrate_cmd':    bcol('substrate_cmd'),
            'sensor_shutter':   bcol('sensor_shutter'),
            'heater_temp':      fcol('heater_temp'),
            'heater_power':     fcol('heater_power'),
            'heater_ref_tc':    fcol('heater_ref_tc'),
            'heater_cal_tc':    fcol('heater_cal_tc'),
            'pyro_temp':        fcol('pyro_temp'),
            'pyro_uncorr':      fcol('pyro_uncorr'),
            'reflectivity':     fcol('reflectivity'),
            'sensor_rate':      fcol('sensor_rate'),
            'sensor_thickness': fcol('sensor_thickness'),
            'tooling':          fcol('tooling'),
            'density':          fcol('density'),
            'zfactor':          fcol('zfactor'),
            'crystal_remaining': fcol('crystal_remaining'),
            'rotation':         fcol('rotation'),
            'z_position':       fcol('z_position'),
            'cal_rate':         fcol('cal_rate'),
            'cal_power':        fcol('cal_power'),
            'cal_est_thickness': fcol('cal_est_thickness'),
            'cal_elapsed':      fcol('cal_elapsed'),
        }

        _cache[run_id] = data
        return data

    except Exception as e:
        print(f'Error loading run {run_id}: {e}')
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Phase name extraction from recipe XML
# ---------------------------------------------------------------------------

def extract_phase_names(xml_path):
    """Extract phase names from a recipe XML, returning {phase_num: name}."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        names = {}
        for layer_el in root.findall(f'.//{{{NS["r"]}}}RecipeLayer'):
            layer_num = int(layer_el.findtext(f'{{{NS["r"]}}}LayerNumber', '0'))
            if layer_num != 1:
                continue
            for phase_el in layer_el.findall(
                    f'{{{NS["r"]}}}Phases/{{{NS["r"]}}}RecipePhase'):
                pnum = int(phase_el.findtext(f'{{{NS["r"]}}}PhaseNumber', '0'))
                pname = phase_el.findtext(f'{{{NS["r"]}}}PhaseName', '')
                if pname:
                    names[pnum] = pname
        return names if names else None
    except Exception:
        return None


def get_phase_names(run_id):
    """Get phase names for a run, trying recipe XML first, then defaults."""
    files = get_run_files(run_id)
    xml_path = files.get('complete_xml') or files.get('original_xml')
    if xml_path:
        names = extract_phase_names(xml_path)
        if names:
            return names
    return PHASE_NAMES.copy()


def extract_recipe_name(xml_path):
    """Extract recipe name from a recipe XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return root.findtext(f'{{{NS["r"]}}}RecipeName', '')
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# Run scanning
# ---------------------------------------------------------------------------

def scan_runs():
    """Scan LOG_DIR for CSV files and return a sorted list of run dicts."""
    runs = []
    for name in os.listdir(LOG_DIR):
        full = os.path.join(LOG_DIR, name)
        if not os.path.isfile(full):
            continue

        m = CSV_RE.search(name)
        if not m:
            continue

        run_id = m.group(1)
        ds, ts = run_id[:8], run_id[9:]
        date = f'{ds[:4]}-{ds[4:6]}-{ds[6:8]}'
        time_s = f'{ts[:2]}:{ts[2:4]}:{ts[4:6]}'

        # Check completeness from last line
        complete = False
        last = read_last_csv_line(full)
        cols = last.split(',')
        if len(cols) > 7:
            lv = cols[5].strip()
            step = cols[7].strip()
            complete = lv in ('-1',) or step == 'COMPLETE'

        # Extract recipe name from CSV filename
        # Pattern: ..._CDep_R<recipe_name>_YYYYMMDD_HHMMSS.csv
        recipe_name = ''
        files = get_run_files(run_id)
        xml_path = files.get('complete_xml') or files.get('original_xml')
        if xml_path:
            recipe_name = extract_recipe_name(xml_path)

        runs.append({
            'id': run_id,
            'date': date,
            'time': time_s,
            'recipe': recipe_name,
            'complete': complete,
        })

    runs.sort(key=lambda r: r['id'], reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(run_id):
    """Compute phase-by-phase summary for a single run."""
    d = load_run(run_id)
    if not d:
        return None

    t      = d['t']
    layer  = d['layer']
    phase  = d['phase']

    phase_names = get_phase_names(run_id)
    phases_out = []

    # Only process Layer 1 phases (the single process layer)
    for P in sorted(set(phase[(layer == 1)])):
        if P == 0:
            continue  # skip transition phases

        mask = (layer == 1) & (phase == P)
        if not mask.any():
            continue

        idx = np.where(mask)[0]
        t_start = t[idx[0]]
        t_end = t[idx[-1]]
        duration = t_end - t_start

        pname = phase_names.get(P, f'Phase {P}')
        entry = {
            'phase': int(P),
            'name': pname,
            'duration': round(duration, 1),
        }

        # -- Heating phases (1, 2) -------------------------------------------
        if P in (1, 2):
            hp = d['heater_power'][mask]
            ht = d['heater_temp'][mask]
            entry['heater_power_mean'] = round(float(np.mean(hp)), 1)
            entry['heater_power_max'] = round(float(np.max(hp)), 1)
            entry['temp_start'] = round(float(ht[0]), 1)
            entry['temp_end'] = round(float(ht[-1]), 1)
            entry['temp_max'] = round(float(np.max(ht)), 1)
            # Pyrometry temp at end (may be zero if not yet active)
            pyro = d['pyro_temp'][idx[-1]]
            entry['pyro_temp_end'] = round(float(pyro), 1) if pyro > 0 else None

        # -- Sputtering phases (3, 4, 5) ------------------------------------
        if P in (3, 4, 5):
            entry['sputter_power'] = round(float(np.mean(d['dc_power'][mask])), 1)
            entry['sputter_voltage'] = round(float(np.mean(d['dc_voltage'][mask])), 1)
            entry['sputter_current'] = round(float(np.mean(d['dc_current'][mask])), 1)
            pf = d['pulse_freq'][mask]
            mean_pf = float(np.mean(pf))
            if mean_pf > 0:
                entry['dc_mode'] = 'Pulsed DC'
                entry['pulse_freq_hz'] = round(mean_pf, 0)
            else:
                entry['dc_mode'] = 'DC'
            # Phase 5 uses calibrated rate (sensor shutter closed, power-based
            # rate control), phases 3-4 use direct source rate
            if P == 5:
                entry['rate_mean'] = round(float(np.mean(d['cal_rate'][mask])), 2)
                entry['thickness_final'] = round(float(d['cal_est_thickness'][idx[-1]]), 1)
            else:
                entry['rate_mean'] = round(float(np.mean(d['src_rate'][mask])), 2)
                entry['thickness_final'] = round(float(d['src_thickness'][idx[-1]]), 1)
            entry['ar_flow'] = round(float(np.mean(d['ar_flow'][mask])), 1)
            entry['heater_power_mean'] = round(float(np.mean(d['heater_power'][mask])), 1)
            # Substrate temp: prefer pyrometry if available
            pyro_vals = d['pyro_temp'][mask]
            pyro_valid = pyro_vals[pyro_vals > 0]
            if len(pyro_valid) > 0:
                entry['substrate_temp'] = round(float(np.mean(pyro_valid)), 1)
            else:
                entry['substrate_temp'] = round(float(np.mean(d['heater_temp'][mask])), 1)
            # Spark count delta
            sparks = d['spark_count'][mask]
            entry['spark_count_delta'] = int(sparks[-1] - sparks[0])

        # -- Cooldown / postcondition phases (6, 7) -------------------------
        if P in (6, 7):
            hp = d['heater_power'][mask]
            ht = d['heater_temp'][mask]
            entry['heater_power_mean'] = round(float(np.mean(hp)), 1)
            entry['temp_start'] = round(float(ht[0]), 1)
            entry['temp_end'] = round(float(ht[-1]), 1)

        phases_out.append(entry)

    # -- Run info -----------------------------------------------------------
    run_date = f'{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}'
    total_duration = float(t[-1]) if len(t) > 0 else 0.0

    # Base pressure: min valid pressure reading
    p_all = d['pressure']
    valid_p = p_all[~np.isnan(p_all)]
    valid_p = valid_p[valid_p > 0]
    base_p = float(np.min(valid_p)) if len(valid_p) > 0 else None

    # Recipe name and material
    files = d['files']
    recipe_name = ''
    xml_path = files.get('complete_xml') or files.get('original_xml')
    if xml_path:
        recipe_name = extract_recipe_name(xml_path)

    return {
        'phases': phases_out,
        'run_info': {
            'date': run_date,
            'total_duration': round(total_duration, 1),
            'base_pressure': base_p,
            'recipe_name': recipe_name,
            'material': 'Tantalum',
        },
    }


# ---------------------------------------------------------------------------
# Timeseries
# ---------------------------------------------------------------------------

def _mask_stale_cal_thickness(cal_thick, shutter, layer, phase):
    """Zero out calibrated thickness outside the active deposition window.
    The column carries a stale value from initialization except during
    the brief phase-5 deposition with substrate shutter open."""
    out = cal_thick.copy()
    active = shutter & (layer == 1) & (phase == 5)
    out[~active] = np.nan
    return out


def get_timeseries(run_id):
    """Return columnar arrays and phase boundaries for charting."""
    d = load_run(run_id)
    if not d:
        return None

    t     = d['t']
    layer = d['layer']
    phase = d['phase']

    phase_names = get_phase_names(run_id)

    # Phase boundaries: every layer or phase transition
    boundaries = []
    for i in range(1, len(layer)):
        if layer[i] != layer[i - 1] or phase[i] != phase[i - 1]:
            L, P = int(layer[i]), int(phase[i])
            if L == 1:
                pname = phase_names.get(P, f'Phase {P}')
                label = f'L1: {pname}'
            elif L == 0:
                label = 'Startup' if P > 0 else 'Init'
            elif L == -1:
                label = 'End'
            else:
                label = f'Layer {L} Phase {P}'
            boundaries.append({
                't':     round(float(t[i]), 2),
                'layer': L,
                'phase': P,
                'label': label,
            })

    return {
        't':               clean(t),
        'layer':           layer.tolist(),
        'phase':           phase.tolist(),
        'shutter':         d['substrate_shutter'].tolist(),
        'src_shutter':     d['src_shutter'].tolist(),
        'sensor_shutter':  d['sensor_shutter'].tolist(),
        'heater_temp':     clean(d['heater_temp']),
        'heater_power':    clean(d['heater_power']),
        'pyro_temp':       clean(d['pyro_temp']),
        'reflectivity':    clean(d['reflectivity']),
        'sputter_power_w': clean(d['dc_power']),
        'sputter_voltage': clean(d['dc_voltage']),
        'sputter_current': clean(d['dc_current']),
        'rate':            clean(d['src_rate']),
        'thickness':       clean(d['src_thickness']),
        'pressure':        clean(d['pressure']),
        'ar_flow':         clean(d['ar_flow']),
        'rotation':        clean(d['rotation']),
        'z_position':      clean(d['z_position']),
        'spark_count':     clean(d['spark_count']),
        'cal_rate':        clean(d['cal_rate']),
        'cal_power':       clean(d['cal_power']),
        'cal_thickness':   clean(_mask_stale_cal_thickness(
            d['cal_est_thickness'], d['substrate_shutter'], d['layer'], d['phase'])),
        'phase_boundaries': boundaries,
    }


# ---------------------------------------------------------------------------
# Recipe XML parsing & diff
# ---------------------------------------------------------------------------

# Which child elements to extract from each action type
PARAM_FIELDS = {
    'a:RecipeActionDepositPowerCalibratedRate':
        ['TargetThickness', 'Timeout'],
    'a:RecipeActionCalibrateRate':
        ['DelayCalibrationStartTime', 'SampleTime'],
    'a:RecipeActionRampPower':
        ['TargetPower', 'RampRate', 'SoakTime'],
    'a:RecipeActionDelay':
        ['DelayTime'],
    'a:RecipeActionProcessPressure':
        ['PressureSetpoint'],
    'a:RecipeActionProcessPressureStatic':
        ['PressureSetpoint'],
    'a:RecipeActionWaitForChamberPressure':
        ['PressureSetpoint'],
    'a:RecipeActionGasFlow':
        ['FlowRate'],
    'a:RecipeActionHeaterTemperature':
        ['TargetTemperature', 'HoldTime', 'RampRate', 'Accuracy'],
    'a:RecipeActionChamberPumpDown':
        ['ChamberNumber', 'EnableRapidPumpdown'],
    'a:RecipeActionConfigurePulsedDCSupply':
        ['ControlMode', 'OperationMode', 'PulseFrequency', 'PulseDuration'],
    'a:RecipeActionSubstrateRotation':
        ['EnableRotation', 'TargetVelocity'],
    'a:RecipeActionControlTempSensorAcquisition':
        ['EnableAcquisition', 'Reflectance', 'TemperatureCrossOver'],
    'a:RecipeActionDepositRate':
        ['TargetThickness', 'TargetRate', 'Timeout'],
    'a:RecipeActionStabilizeRate':
        ['TargetSetpoint', 'HoldTime', 'Accuracy', 'Timeout'],
}


def extract_recipe_params(xml_path):
    """Parse a recipe XML and return a flat {path: value} dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    params = {}

    def process_phase(layer_num, layer_name, phase_el):
        phase_name = phase_el.findtext(f'{{{NS["r"]}}}PhaseName', '')
        prefix = f'Layer {layer_num} / {layer_name} / {phase_name}'

        for step in phase_el.findall(f'{{{NS["r"]}}}Steps/{{{NS["r"]}}}RecipeStep'):
            for action in step.findall(
                    f'{{{NS["r"]}}}Actions/{{{NS["a"]}}}RecipeActionBaseClass'):
                atype = action.get(f'{{{NS["i"]}}}type', '')

                # Standard scalar fields
                fields = PARAM_FIELDS.get(atype, [])
                for field in fields:
                    el = action.find(f'{{{NS["a"]}}}{field}')
                    if el is not None and el.text is not None:
                        params[f'{prefix} / {field}'] = el.text

    # Process recipe layers
    for layer_el in root.findall(f'.//{{{NS["r"]}}}RecipeLayer'):
        layer_num = int(layer_el.findtext(f'{{{NS["r"]}}}LayerNumber', '0'))
        layer_name = layer_el.findtext(f'{{{NS["r"]}}}LayerName', '')
        for phase_el in layer_el.findall(
                f'{{{NS["r"]}}}Phases/{{{NS["r"]}}}RecipePhase'):
            process_phase(layer_num, layer_name, phase_el)

    # Startup and Cleanup are top-level, not inside <Layers>
    startup = root.find(f'{{{NS["r"]}}}Startup')
    if startup is not None:
        process_phase(0, 'Startup', startup)

    cleanup = root.find(f'{{{NS["r"]}}}Cleanup')
    if cleanup is not None:
        process_phase(2, 'Cleanup', cleanup)

    return params


def compute_recipe_diff(run_id):
    """Compare Original vs Complete recipe XMLs for a run."""
    files = get_run_files(run_id)
    orig_path = files.get('original_xml')
    comp_path = files.get('complete_xml')

    if not orig_path or not comp_path:
        return {'has_changes': False, 'changes': [], 'all_params': []}

    try:
        orig = extract_recipe_params(orig_path)
        comp = extract_recipe_params(comp_path)
    except Exception as e:
        print(f'Error parsing recipe XML for {run_id}: {e}')
        return {'has_changes': False, 'changes': [], 'all_params': []}

    all_keys = sorted(set(orig.keys()) | set(comp.keys()))
    changes    = []
    all_params = []

    for key in all_keys:
        ov = orig.get(key)
        cv = comp.get(key)
        val = cv if cv is not None else ov
        all_params.append({'path': key, 'value': val})
        if ov != cv:
            changes.append({'path': key, 'original': ov, 'complete': cv})

    return {
        'has_changes': len(changes) > 0,
        'changes':     changes,
        'all_params':  all_params,
    }


# ---------------------------------------------------------------------------
# Cross-run comparison
# ---------------------------------------------------------------------------

def compute_comparison(run_ids=None):
    """Aggregate key metrics across selected runs for trend analysis."""
    all_runs = scan_runs()
    if run_ids:
        id_set = set(run_ids)
        all_runs = [r for r in all_runs if r['id'] in id_set]

    result_runs = []
    result_ids  = []

    metrics = {
        'dep_rate':         {'values': [],                'unit': '\u00c5/s'},
        'thickness':        {'values': [],                'unit': '\u00c5'},
        'sputter_power':    {'values': [],                'unit': 'W'},
        'sputter_voltage':  {'values': [],                'unit': 'V'},
        'sputter_current':  {'values': [],                'unit': 'mA'},
        'ar_flow':          {'values': [],                'unit': 'sccm'},
        'substrate_temp':   {'values': [],                'unit': '\u00b0C'},
        'heater_power':     {'values': [],                'unit': '%'},
        'spark_count':      {'values': [],                'unit': ''},
        'base_pressure':    {'values': [],                'unit': 'Torr'},
        'crystal_remaining':{'values': [],                'unit': '%'},
    }

    # Process runs in chronological order
    for run in sorted(all_runs, key=lambda r: r['id']):
        run_id = run['id']
        summary = compute_summary(run_id)
        if not summary:
            continue

        result_runs.append(run['date'])
        result_ids.append(run_id)

        # Find deposition phase (phase 5)
        dep_phase = None
        for p in summary['phases']:
            if p['phase'] == 5:
                dep_phase = p
                break

        metrics['dep_rate']['values'].append(
            dep_phase.get('rate_mean') if dep_phase else None)
        metrics['thickness']['values'].append(
            dep_phase.get('thickness_final') if dep_phase else None)
        metrics['sputter_power']['values'].append(
            dep_phase.get('sputter_power') if dep_phase else None)
        metrics['sputter_voltage']['values'].append(
            dep_phase.get('sputter_voltage') if dep_phase else None)
        metrics['sputter_current']['values'].append(
            dep_phase.get('sputter_current') if dep_phase else None)
        metrics['ar_flow']['values'].append(
            dep_phase.get('ar_flow') if dep_phase else None)
        metrics['substrate_temp']['values'].append(
            dep_phase.get('substrate_temp') if dep_phase else None)
        metrics['heater_power']['values'].append(
            dep_phase.get('heater_power_mean') if dep_phase else None)
        metrics['spark_count']['values'].append(
            dep_phase.get('spark_count_delta') if dep_phase else None)

        # Base pressure
        metrics['base_pressure']['values'].append(
            summary['run_info']['base_pressure'])

        # Crystal life remaining (last positive value)
        d = load_run(run_id)
        if d is not None and len(d['crystal_remaining']) > 0:
            cr_vals = d['crystal_remaining']
            cr_pos = cr_vals[cr_vals > 0]
            if len(cr_pos) > 0:
                metrics['crystal_remaining']['values'].append(
                    round(float(cr_pos[-1]), 1))
            else:
                metrics['crystal_remaining']['values'].append(None)
        else:
            metrics['crystal_remaining']['values'].append(None)

    return {
        'runs':    result_runs,
        'run_ids': result_ids,
        'metrics': metrics,
    }


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/')
def index():
    return send_from_directory(APP_DIR, 'ta-index.html')


@app.route('/api/runs')
def api_runs():
    return jsonify(scan_runs())


@app.route('/api/run/<run_id>/summary')
def api_summary(run_id):
    result = compute_summary(run_id)
    if result is None:
        abort(404)
    return jsonify(result)


@app.route('/api/run/<run_id>/timeseries')
def api_timeseries(run_id):
    result = get_timeseries(run_id)
    if result is None:
        abort(404)
    return jsonify(result)


@app.route('/api/run/<run_id>/recipe-diff')
def api_recipe_diff(run_id):
    result = compute_recipe_diff(run_id)
    if result is None:
        abort(404)
    return jsonify(result)


@app.route('/api/run/<run_id>/valves')
def api_valves(run_id):
    d = load_run(run_id)
    if d is None:
        abort(404)

    t     = d['t']
    layer = d['layer']
    phase = d['phase']

    phase_names = get_phase_names(run_id)

    boundaries = []
    for i in range(1, len(layer)):
        if layer[i] != layer[i - 1] or phase[i] != phase[i - 1]:
            L, P = int(layer[i]), int(phase[i])
            if L == 1:
                pname = phase_names.get(P, f'Phase {P}')
                label = f'L1: {pname}'
            elif L == 0:
                label = 'Startup' if P > 0 else 'Init'
            elif L == -1:
                label = 'End'
            else:
                label = f'Layer {L} Phase {P}'
            boundaries.append({
                't': round(float(t[i]), 2),
                'layer': L, 'phase': P,
                'label': label,
            })

    return jsonify({
        't':                 clean(t),
        'layer':             layer.tolist(),
        'phase':             phase.tolist(),
        'substrate_shutter': d['substrate_shutter'].tolist(),
        'substrate_cmd':     d['substrate_cmd'].tolist(),
        'source_shutter':    d['src_shutter'].tolist(),
        'sensor_shutter':    d['sensor_shutter'].tolist(),
        'phase_boundaries':  boundaries,
    })


@app.route('/api/comparison')
def api_comparison():
    run_ids = request.args.get('runs', '')
    run_ids = [r.strip() for r in run_ids.split(',') if r.strip()] or None
    return jsonify(compute_comparison(run_ids))


@app.route('/api/config', methods=['GET'])
def api_config():
    return jsonify({'log_dir': LOG_DIR})


@app.route('/api/config', methods=['POST'])
def api_set_config():
    data = request.get_json(force=True)
    new_dir = data.get('log_dir', '').strip()
    if not new_dir or not os.path.isdir(new_dir):
        return jsonify({'error': 'Directory does not exist'}), 400
    global LOG_DIR
    LOG_DIR = new_dir
    _cache.clear()
    return jsonify({'log_dir': LOG_DIR})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Angstrom Ta Sputterer Log Viewer',
        epilog='''
Expected directory layout (FLAT — no subfolders):
  <log-dir>/
    <SampleName>_<RecipeName>_YYYYMMDD_HHMMSS.csv
    <SampleName>_<RecipeName>_YYYYMMDD_HHMMSS_details.json
    <RecipeName>_Original_<timestamp>.xml
    <RecipeName>_Complete_<timestamp>.xml
    <RecipeName>_Status_<timestamp>.xml
    Hatlab_*_<timestamp>.xml   (Nebula cluster tool — ignored)

Each CSV is one run. The run_id is the YYYYMMDD_HHMMSS timestamp
extracted from the CSV filename.
''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--log-dir', default=LOG_DIR,
        help='Path to directory containing log files (default: %(default)s)')
    parser.add_argument(
        '--port', type=int, default=5001,
        help='Port to serve on (default: 5001)')
    args = parser.parse_args()

    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        print(f'Error: log directory does not exist: {LOG_DIR}')
        raise SystemExit(1)

    print('Angstrom Ta Sputterer Log Viewer')
    print(f'  Log directory : {LOG_DIR}')
    print(f'  Serving       : http://localhost:{args.port}')
    app.run(host='0.0.0.0', port=args.port, debug=False)
