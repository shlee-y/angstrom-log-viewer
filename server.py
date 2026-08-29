"""
Flask backend for Angstrom JJ evaporator log viewer.
Serves CSV time-series data, layer summaries, recipe diffs,
and cross-run comparisons for Angstrom EvoVac JJ evaporator runs.
"""

import os
import re
import json
import math
import glob as globmod

from flask import Flask, jsonify, send_from_directory, abort
from flask.json.provider import DefaultJSONProvider
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, 'Angstrom log')
FOLDER_RE = re.compile(r'CQtm_.*_(\d{8}_\d{6})$')

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

LAYER_NAMES = {
    0: 'Startup', 1: 'Ar etch', 2: 'Ti gettering',
    3: 'Al 1st', 4: 'Junction Oxidation', 5: 'Al 2nd',
    6: 'Capping Oxidation', 7: 'Cleanup',
}

PHASE_NAMES = {
    0: {0: 'Init', 1: 'Startup'},
    1: {1: 'Purge', 2: 'ConfigKDC', 3: 'Discharge',
        4: 'Etch', 5: 'Cooldown', 6: 'PumpDown'},
    2: {1: 'VAD', 2: 'Precondition', 3: 'Stabilize',
        4: 'Deposit', 5: 'PostCondition'},
    3: {1: 'VAD', 2: 'Precondition', 3: 'Stabilize',
        4: 'Deposit', 5: 'PostCondition'},
    4: {1: 'Oxidation', 2: 'PumpDown'},
    5: {1: 'VAD', 2: 'Precondition', 3: 'Stabilize',
        4: 'Deposit', 5: 'PostCondition'},
    6: {1: 'Oxidation', 2: 'PumpDown'},
    7: {1: 'Transfer'},
}

DEPOSITION_LAYERS = {
    2: {'material': 'Titanium',  'thickness': 600, 'rate': 2.0, 'source': 'ti'},
    3: {'material': 'Aluminum',  'thickness': 200, 'rate': 5.0, 'source': 'al'},
    5: {'material': 'Aluminum',  'thickness': 730, 'rate': 5.0, 'source': 'al'},
}

OXIDATION_LAYERS = {
    4: {'pressure': 30.0,  'duration': 600},
    6: {'pressure': 9.75,  'duration': 300},
}

# XML namespaces used in the Angstrom recipe files
NS = {
    'r': 'http://schemas.datacontract.org/2004/07/AE.DepControl.DepositionRecipe',
    'a': 'http://schemas.datacontract.org/2004/07/AE.DepControl.DepositionRecipe.Actions',
    'i': 'http://www.w3.org/2001/XMLSchema-instance',
    'v': 'http://schemas.datacontract.org/2004/07/AE.DepControl.DepositionRecipe.Actions.Vad.Moves',
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


def find_file(folder, pattern):
    """Return first file matching *pattern* inside *folder*, or None."""
    matches = globmod.glob(os.path.join(folder, pattern))
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

    The _details.json has 254 entries corresponding to CSV columns 5-258
    (columns 0-4 are the logger's own: Version, Date, Time, Elapsed Time,
    Triggered).  We use the *Name* field, keeping the first occurrence for
    names that repeat across source blocks.
    """
    with open(details_path, encoding='utf-8') as f:
        details = json.load(f)

    n2c = {}
    for i, entry in enumerate(details):
        name = entry['Name']
        if name not in n2c:
            n2c[name] = i + 5          # offset for the 5 logger columns

    m = {
        'layer':         n2c['DepSequencer.CurrentLayer'],
        'phase':         n2c['DepSequencer.CurrentPhase'],
        'step':          n2c['DepSequencer.StepName'],
        'ox_pressure':   n2c['Chamber[0].ProcessPressure'],
        'ox_chamber':    n2c['Chamber[0].Pressure'],
        'evap_pressure': n2c['Chamber[2].Pressure'],
        'shutter':       n2c['SeqSubstrate[0].ShutterOpen'],
        'shutter_cmd':   n2c['SeqSubstrate[0].bOpenShutter'],
        'rotation':      n2c['Servo[3].stActual.Position'],
        'tilt':          n2c['Servo[4].stActual.Position'],
        'ti_thickness':  n2c['Source[2].ActualThickness'],
        'ti_rate':       n2c['Source[2].ActualRate'],
        'ti_power':      n2c['Source[2].ActualPower'],
        'ti_beam':       n2c['EbeamSupply[2].BeamCurrent'],
        'al_thickness':  n2c['Source[3].ActualThickness'],
        'al_rate':       n2c['Source[3].ActualRate'],
        'al_power':      n2c['Source[3].ActualPower'],
        'al_beam':       n2c['EbeamSupply[3].BeamCurrent'],
        'mc1_rate':      n2c['Sensor[1].rRate'],
        'mc1_remaining': n2c['PhysicalSensor[1].rPercentRemaining'],
        'dual_rate':     n2c['PhysicalSensor[2].rRate'],
        'kdc_beam_v':    n2c['KDC[5].BeamVoltage'],
        'kdc_beam_i':    n2c['KDC[5].BeamCurrent'],
        'kdc_ar_flow':   n2c['KDC[5].GasChannel1Flow'],
        'mfc3_flow':     n2c['MFC[3].ActualFlow'],
        'ti_src_shutter':  n2c['Source[2].OpenShutter'],
        'al_src_shutter':  n2c['Source[3].OpenShutter'],
        'kdc_shutter':     n2c['Source[5].OpenShutter'],
        'mc1_shutter':     n2c['Sensor[1].bOpenShutter'],
        'dual_shutter':    n2c['Sensor[2].bOpenShutter'],
        'unif_shutter':    n2c['Servo[2].stActual.Position'],
        'mc1_tooling':     n2c['PhysicalSensor[1].ToolingFactor'],
        'mc1_density':     n2c['PhysicalSensor[1].Density'],
        'mc1_zfactor':     n2c['PhysicalSensor[1].ZFactor'],
        'mc1_crystal':     n2c['Sensor[1].CrystalIndexerCurrentUIIndex'],
        'dual_tooling':    n2c['PhysicalSensor[2].ToolingFactor'],
        'dual_density':    n2c['PhysicalSensor[2].Density'],
        'dual_zfactor':    n2c['PhysicalSensor[2].ZFactor'],
    }
    return m


# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------

def get_run_folder(run_id):
    """Return the full path to a run folder given its timestamp id."""
    for name in os.listdir(LOG_DIR):
        if name.endswith(run_id) and FOLDER_RE.match(name):
            return os.path.join(LOG_DIR, name)
    return None


def load_run(run_id):
    """Load a run's CSV data, parse the columns we need, and cache the result.

    Returns a dict of numpy arrays keyed by channel name, or None on error.
    """
    if run_id in _cache:
        return _cache[run_id]

    folder = get_run_folder(run_id)
    if not folder:
        return None

    csv_path = find_file(folder, '*.csv')
    det_path = find_file(folder, '*_details.json')
    if not csv_path or not det_path:
        return None

    try:
        col = build_col_map(det_path)
        use_cols = sorted(set([3] + list(col.values())))

        df = pd.read_csv(csv_path, header=None, skiprows=1,
                         usecols=use_cols, low_memory=False)

        t = df[3].apply(parse_elapsed).values

        layer = pd.to_numeric(df[col['layer']], errors='coerce') \
                  .fillna(0).astype(int).values
        phase = pd.to_numeric(df[col['phase']], errors='coerce') \
                  .fillna(0).astype(int).values
        shutter = (df[col['shutter']].astype(str).str.strip() == 'True').values

        def fcol(key):
            return pd.to_numeric(df[col[key]], errors='coerce') \
                     .fillna(0.0).values.copy()

        def pcol(key):
            v = pd.to_numeric(df[col[key]], errors='coerce').values.copy()
            v[v <= -9000] = np.nan
            return v

        data = {
            'folder':        folder,
            't':             t,
            'layer':         layer,
            'phase':         phase,
            'shutter':       shutter,
            'ti_rate':       fcol('ti_rate'),
            'ti_thickness':  fcol('ti_thickness'),
            'ti_power':      fcol('ti_power'),
            'ti_beam':       fcol('ti_beam'),
            'al_rate':       fcol('al_rate'),
            'al_thickness':  fcol('al_thickness'),
            'al_power':      fcol('al_power'),
            'al_beam':       fcol('al_beam'),
            'mc1_rate':      fcol('mc1_rate'),
            'mc1_remaining': fcol('mc1_remaining'),
            'dual_rate':     fcol('dual_rate'),
            'evap_pressure': pcol('evap_pressure'),
            'ox_pressure':   pcol('ox_pressure'),
            'ox_chamber':    pcol('ox_chamber'),
            'tilt':          fcol('tilt'),
            'rotation':      fcol('rotation'),
            'kdc_beam_v':    fcol('kdc_beam_v'),
            'kdc_beam_i':    fcol('kdc_beam_i'),
            'kdc_ar_flow':   fcol('kdc_ar_flow'),
            'mfc3_flow':     fcol('mfc3_flow'),
            'shutter_cmd':       (df[col['shutter_cmd']].astype(str).str.strip() == 'True').values,
            'ti_src_shutter':    (df[col['ti_src_shutter']].astype(str).str.strip() == 'True').values,
            'al_src_shutter':    (df[col['al_src_shutter']].astype(str).str.strip() == 'True').values,
            'kdc_shutter':       (df[col['kdc_shutter']].astype(str).str.strip() == 'True').values,
            'mc1_shutter':       (df[col['mc1_shutter']].astype(str).str.strip() == 'True').values,
            'dual_shutter':      (df[col['dual_shutter']].astype(str).str.strip() == 'True').values,
            'unif_shutter':      fcol('unif_shutter'),
            'mc1_tooling':       fcol('mc1_tooling'),
            'mc1_density':       fcol('mc1_density'),
            'mc1_zfactor':       fcol('mc1_zfactor'),
            'mc1_crystal':       fcol('mc1_crystal'),
            'dual_tooling':      fcol('dual_tooling'),
            'dual_density':      fcol('dual_density'),
            'dual_zfactor':      fcol('dual_zfactor'),
        }

        _cache[run_id] = data
        return data

    except Exception as e:
        print(f'Error loading run {run_id}: {e}')
        return None


# ---------------------------------------------------------------------------
# Run scanning
# ---------------------------------------------------------------------------

def scan_runs():
    """Scan LOG_DIR for run folders and return a sorted list of run dicts."""
    runs = []
    for name in os.listdir(LOG_DIR):
        path = os.path.join(LOG_DIR, name)
        if not os.path.isdir(path):
            continue
        m = FOLDER_RE.match(name)
        if not m:
            continue

        run_id = m.group(1)
        ds, ts = run_id[:8], run_id[9:]
        date = f'{ds[:4]}-{ds[4:6]}-{ds[6:8]}'
        time_s = f'{ts[:2]}:{ts[2:4]}:{ts[4:6]}'

        csv_path = find_file(path, '*.csv')
        complete = False
        if csv_path:
            last = read_last_csv_line(csv_path)
            cols = last.split(',')
            if len(cols) > 7:
                lv = cols[5].strip()
                step = cols[7].strip()
                complete = lv in ('7', '-1') or step == 'COMPLETE'

        runs.append({
            'id': run_id,
            'date': date,
            'time': time_s,
            'folder': name,
            'complete': complete,
        })

    runs.sort(key=lambda r: r['id'], reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(run_id):
    """Compute layer-by-layer summary for a single run."""
    d = load_run(run_id)
    if not d:
        return None

    t      = d['t']
    layer  = d['layer']
    phase  = d['phase']
    shutter = d['shutter']

    layers_out = []

    def get_angles(mask):
        """Get median tilt and rotation for a given mask."""
        if not mask.any():
            return None, None
        tilt_v = float(np.median(d['tilt'][mask]))
        rot_v = float(np.median(d['rotation'][mask]))
        # Normalize 360° → 0°
        if abs(rot_v - 360) < 1:
            rot_v = 0.0
        return round(tilt_v, 1), round(rot_v, 1)

    # -- Ar etch (layer 1) ---------------------------------------------------
    etch_mask = (layer == 1) & (phase == 4)
    if etch_mask.any():
        etch_idx = np.where(etch_mask)[0]
        etch_duration = t[etch_idx[-1]] - t[etch_idx[0]]
        beam_voltage  = float(np.mean(d['kdc_beam_v'][etch_mask]))
        beam_current  = float(np.mean(d['kdc_beam_i'][etch_mask]))
        ar_flow       = float(np.mean(d['kdc_ar_flow'][etch_mask]))
        tilt, rot = get_angles(etch_mask)

        layers_out.append({
            'layer': 1,
            'name': 'Ar etch',
            'type': 'etch',
            'etch_duration': round(etch_duration, 1),
            'beam_voltage': round(beam_voltage, 1),
            'beam_current': round(beam_current, 2),
            'ar_flow': round(ar_flow, 1),
            'tilt': tilt,
            'rotation': rot,
        })

    # -- Deposition layers (2, 3, 5) ----------------------------------------
    for L, info in sorted(DEPOSITION_LAYERS.items()):
        src = info['source']               # 'ti' or 'al'
        rate_arr  = d[f'{src}_rate']
        thick_arr = d[f'{src}_thickness']
        power_arr = d[f'{src}_power']

        layer_mask = layer == L
        if not layer_mask.any():
            continue

        # Gate on substrate shutter open during the Deposit phase
        dep_mask = layer_mask & (phase == 4) & shutter
        if not dep_mask.any():
            continue

        dep_idx  = np.where(dep_mask)[0]
        t_start  = t[dep_idx[0]]
        t_end    = t[dep_idx[-1]]

        rates      = rate_arr[dep_mask]
        rates_dual = d['dual_rate'][dep_mask]
        powers     = power_arr[dep_mask]
        thickness  = float(thick_arr[dep_idx[-1]])

        # Max evaporation pressure during deposition (skip NaN)
        evap_p  = d['evap_pressure'][dep_mask]
        valid_p = evap_p[~np.isnan(evap_p)]
        p_max   = float(np.max(valid_p)) if len(valid_p) > 0 else None

        # Rate at end of Precondition phase (phase 2)
        precond_mask = layer_mask & (phase == 2)
        precond_end_rate = None
        if precond_mask.any():
            precond_end_rate = float(rate_arr[np.where(precond_mask)[0][-1]])

        tilt, rot = get_angles(dep_mask)

        # Sensor calibration (mid-deposit values)
        mid = dep_idx[len(dep_idx) // 2]
        cal = {
            'mc1_tooling':  round(float(d['mc1_tooling'][mid]), 2),
            'mc1_density':  round(float(d['mc1_density'][mid]), 2),
            'mc1_zfactor':  round(float(d['mc1_zfactor'][mid]), 3),
            'mc1_crystal':  int(d['mc1_crystal'][mid]),
            'dual_tooling': round(float(d['dual_tooling'][mid]), 2),
            'dual_density': round(float(d['dual_density'][mid]), 2),
            'dual_zfactor': round(float(d['dual_zfactor'][mid]), 3),
        }

        layers_out.append({
            'layer':            L,
            'name':             LAYER_NAMES[L],
            'material':         info['material'],
            'thickness_target': info['thickness'],
            'thickness_actual': round(thickness, 1),
            'rate_setpoint':    info['rate'],
            'rate_mean':        round(float(np.mean(rates)), 2),
            'rate_std':         round(float(np.std(rates)), 3),
            'rate_mean_dual':   round(float(np.mean(rates_dual)), 2),
            'power_mean':       round(float(np.mean(powers)), 2),
            'shutter_duration': round(float(t_end - t_start), 1),
            'pressure_max':     p_max,
            'precond_end_rate': round(precond_end_rate, 2) if precond_end_rate is not None else None,
            'tilt':             tilt,
            'rotation':         rot,
            'calibration':      cal,
        })

    # -- Oxidation layers (4, 6) --------------------------------------------
    for L, info in sorted(OXIDATION_LAYERS.items()):
        layer_mask = layer == L
        if not layer_mask.any():
            continue

        ox_mask = layer_mask & (phase == 1)
        if not ox_mask.any():
            continue

        ox_idx  = np.where(ox_mask)[0]
        t_start = t[ox_idx[0]]
        t_end   = t[ox_idx[-1]]

        ox_p    = d['ox_pressure'][ox_mask]
        # Filter to steady-state only (above 90% of setpoint) to exclude
        # the ramp-up from vacuum to the target pressure
        valid_p = ox_p[(~np.isnan(ox_p)) & (ox_p > info['pressure'] * 0.9)]

        # Peak flow during the fill (static pressure control — MFC bursts
        # then sits at zero, so mean is misleading; peak is the fill rate)
        mfc_flow = d['mfc3_flow'][ox_mask]
        peak_flow = float(np.max(mfc_flow)) if len(mfc_flow) > 0 else None

        tilt, rot = get_angles(ox_mask)

        layers_out.append({
            'layer':             L,
            'name':              LAYER_NAMES[L],
            'type':              'oxidation',
            'pressure_setpoint': info['pressure'],
            'pressure_actual':   round(float(np.mean(valid_p)), 2) if len(valid_p) > 0 else None,
            'duration':          round(float(t_end - t_start), 1),
            'gas':               'Ar/O₂ 85:15',
            'peak_flow':         round(peak_flow, 1) if peak_flow else None,
            'tilt':              tilt,
            'rotation':          rot,
        })

    layers_out.sort(key=lambda x: x['layer'])

    # -- Run info -----------------------------------------------------------
    run_date = f'{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}'
    total_duration = float(t[-1]) if len(t) > 0 else 0.0

    evap_all = d['evap_pressure']
    valid_evap = evap_all[~np.isnan(evap_all)]
    base_p = float(np.min(valid_evap)) if len(valid_evap) > 0 else None

    return {
        'layers': layers_out,
        'run_info': {
            'date':           run_date,
            'total_duration': round(total_duration, 1),
            'base_pressure':  base_p,
        },
    }


# ---------------------------------------------------------------------------
# Timeseries
# ---------------------------------------------------------------------------

def get_timeseries(run_id):
    """Return columnar arrays and phase boundaries for charting."""
    d = load_run(run_id)
    if not d:
        return None

    t     = d['t']
    layer = d['layer']
    phase = d['phase']

    # Active-source rate from MC1: whichever source is depositing
    rate_mc1 = np.maximum(d['ti_rate'], d['al_rate'])

    # Phase boundaries: every layer or phase transition
    boundaries = []
    for i in range(1, len(layer)):
        if layer[i] != layer[i - 1] or phase[i] != phase[i - 1]:
            L, P = int(layer[i]), int(phase[i])
            lname = LAYER_NAMES.get(L, f'Layer {L}')
            pname = PHASE_NAMES.get(L, {}).get(P, f'Phase {P}')
            boundaries.append({
                't':     round(float(t[i]), 2),
                'layer': L,
                'phase': P,
                'label': f'{lname}: {pname}',
            })

    return {
        't':               clean(t),
        'layer':           layer.tolist(),
        'phase':           phase.tolist(),
        'shutter':         d['shutter'].tolist(),
        'rate_mc1':        clean(rate_mc1),
        'rate_dual':       clean(d['dual_rate']),
        'power_ti':        clean(d['ti_power']),
        'power_al':        clean(d['al_power']),
        'thickness_ti':    clean(d['ti_thickness']),
        'thickness_al':    clean(d['al_thickness']),
        'pressure_evap':   clean(d['evap_pressure']),
        'pressure_ox':     clean(d['ox_pressure']),
        'pressure_ox_ch':  clean(d['ox_chamber']),
        'beam_current_ti': clean(d['ti_beam']),
        'beam_current_al': clean(d['al_beam']),
        'tilt':            clean(d['tilt']),
        'rotation':        clean(d['rotation']),
        'mfc3_flow':       clean(d['mfc3_flow']),
        'kdc_beam_i':      clean(d['kdc_beam_i']),
        'ti_src_shutter':  d['ti_src_shutter'].tolist(),
        'al_src_shutter':  d['al_src_shutter'].tolist(),
        'kdc_shutter':     d['kdc_shutter'].tolist(),
        'mc1_shutter':     d['mc1_shutter'].tolist(),
        'dual_shutter':    d['dual_shutter'].tolist(),
        'phase_boundaries': boundaries,
    }


# ---------------------------------------------------------------------------
# Recipe XML parsing & diff
# ---------------------------------------------------------------------------

# Which child elements to extract from each action type
PARAM_FIELDS = {
    'a:RecipeActionDepositRate':
        ['TargetThickness', 'TargetRate', 'Timeout'],
    'a:RecipeActionStabilizeRate':
        ['TargetSetpoint', 'HoldTime', 'Accuracy', 'Timeout'],
    'a:RecipeActionRampPower':
        ['TargetPower', 'RampRate', 'SoakTime'],
    'a:RecipeActionDelay':
        ['DelayTime'],
    'a:RecipeActionProcessPressureStatic':
        ['PressureSetpoint'],
    'a:RecipeActionWaitForChamberPressure':
        ['PressureSetpoint'],
    'a:RecipeActionGasFlow':
        ['FlowRate'],
    'a:RecipeActionConfigureIonSource_KDC40':
        ['BeamVoltage', 'BeamCurrent', 'DischargeCurrent',
         'DischargeVoltage', 'CathodeCurrent', 'FilamentCurrent',
         'GasChannel1Flow'],
    'a:RecipeActionIonBeamDischargeNoMaterial':
        ['Timeout'],
    'a:RecipeActionTriggerIonSource':
        ['TurnOn'],
    'a:RecipeActionTransferPart':
        ['Timeout'],
    'a:RecipeActionChamberPumpDown':
        ['ChamberNumber', 'EnableRapidPumpdown'],
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

                # VAD motion: extract tilt/rotation from SelectedMoveType
                if atype == 'a:RecipeActionSimpleVADMotion':
                    sel = action.find(f'{{{NS["a"]}}}SelectedMoveType')
                    if sel is not None:
                        ns_v = NS['v']
                        for tag in ('TiltAngle', 'RotationAngle',
                                    'TiltAngleVelocity', 'RotationAngleVelocity'):
                            el = sel.find(f'{{{ns_v}}}{tag}')
                            if el is not None and el.text is not None:
                                params[f'{prefix} / {tag}'] = el.text

    # Process recipe layers
    for layer_el in root.findall(
            f'.//{{{NS["r"]}}}RecipeLayer'):
        layer_num  = int(layer_el.findtext(f'{{{NS["r"]}}}LayerNumber', '0'))
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
        process_phase(7, 'Cleanup', cleanup)

    return params


def compute_recipe_diff(run_id):
    """Compare Original vs Complete recipe XMLs for a run."""
    folder = get_run_folder(run_id)
    if not folder:
        return None

    orig_path = find_file(folder, '*_Original_*.xml')
    comp_path = find_file(folder, '*_Complete_*.xml')

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

def get_etch_time(folder):
    """Extract etch Delay time (seconds) from the Complete recipe XML."""
    xml_path = find_file(folder, '*_Complete_*.xml')
    if not xml_path:
        xml_path = find_file(folder, '*_Original_*.xml')
    if not xml_path:
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for layer_el in root.findall(f'.//{{{NS["r"]}}}RecipeLayer'):
            if layer_el.findtext(f'{{{NS["r"]}}}LayerNumber', '') == '1':
                for phase_el in layer_el.findall(
                        f'{{{NS["r"]}}}Phases/{{{NS["r"]}}}RecipePhase'):
                    if phase_el.findtext(f'{{{NS["r"]}}}PhaseNumber', '') == '4':
                        for action in phase_el.findall(
                                f'.//{{{NS["a"]}}}RecipeActionBaseClass'):
                            atype = action.get(f'{{{NS["i"]}}}type', '')
                            if atype == 'a:RecipeActionDelay':
                                dt = action.findtext(
                                    f'{{{NS["a"]}}}DelayTime', None)
                                if dt is not None:
                                    return float(dt)
    except Exception:
        pass
    return None


def compute_comparison(run_ids=None):
    """Aggregate key metrics across selected runs for trend analysis."""
    all_runs = scan_runs()
    if run_ids:
        id_set = set(run_ids)
        all_runs = [r for r in all_runs if r['id'] in id_set]

    result_runs = []
    result_ids  = []

    metrics = {
        'ti_thickness':        {'values': [], 'target': 600,  'unit': '\u00c5'},
        'ti_rate_mean':        {'values': [], 'target': 2.0,  'unit': '\u00c5/s'},
        'ti_power_mean':       {'values': [],                 'unit': '%'},
        'ti_shutter_duration': {'values': [],                 'unit': 's'},
        'al1_thickness':       {'values': [], 'target': 200,  'unit': '\u00c5'},
        'al1_rate_mean':       {'values': [], 'target': 5.0,  'unit': '\u00c5/s'},
        'al1_rate_dual':       {'values': [],                 'unit': '\u00c5/s'},
        'al1_power_mean':      {'values': [],                 'unit': '%'},
        'al1_shutter_duration':{'values': [],                 'unit': 's'},
        'al2_thickness':       {'values': [], 'target': 730,  'unit': '\u00c5'},
        'al2_rate_mean':       {'values': [], 'target': 5.0,  'unit': '\u00c5/s'},
        'al2_rate_dual':       {'values': [],                 'unit': '\u00c5/s'},
        'al2_power_mean':      {'values': [],                 'unit': '%'},
        'al2_shutter_duration':{'values': [],                 'unit': 's'},
        'jox_pressure':        {'values': [], 'target': 30.0, 'unit': 'Torr'},
        'jox_duration':        {'values': [], 'target': 600,  'unit': 's'},
        'cap_pressure':        {'values': [], 'target': 9.75, 'unit': 'Torr'},
        'cap_duration':        {'values': [], 'target': 300,  'unit': 's'},
        'etch_time':           {'values': [],                 'unit': 's'},
        'etch_beam_v':         {'values': [],                 'unit': 'V'},
        'etch_beam_i':         {'values': [],                 'unit': 'mA'},
        'etch_ar_flow':        {'values': [],                 'unit': 'sccm'},
        'jox_flow':            {'values': [],                 'unit': 'sccm'},
        'cap_flow':            {'values': [],                 'unit': 'sccm'},
        'base_pressure':       {'values': [],                 'unit': 'Torr'},
        'mc1_remaining':       {'values': [],                 'unit': '%'},
    }

    # Process runs in chronological order
    for run in sorted(all_runs, key=lambda r: r['id']):
        run_id = run['id']
        summary = compute_summary(run_id)
        if not summary:
            continue

        result_runs.append(run['date'])
        result_ids.append(run_id)

        # Build quick lookup by layer number
        lmap = {l['layer']: l for l in summary['layers']}

        def val(layer_num, field):
            ly = lmap.get(layer_num)
            return ly.get(field) if ly else None

        # Ti gettering (layer 2)
        metrics['ti_thickness']['values'].append(val(2, 'thickness_actual'))
        metrics['ti_rate_mean']['values'].append(val(2, 'rate_mean'))
        metrics['ti_power_mean']['values'].append(val(2, 'power_mean'))
        metrics['ti_shutter_duration']['values'].append(val(2, 'shutter_duration'))

        # Al 1st (layer 3)
        metrics['al1_thickness']['values'].append(val(3, 'thickness_actual'))
        metrics['al1_rate_mean']['values'].append(val(3, 'rate_mean'))
        metrics['al1_rate_dual']['values'].append(val(3, 'rate_mean_dual'))
        metrics['al1_power_mean']['values'].append(val(3, 'power_mean'))
        metrics['al1_shutter_duration']['values'].append(val(3, 'shutter_duration'))

        # Al 2nd (layer 5)
        metrics['al2_thickness']['values'].append(val(5, 'thickness_actual'))
        metrics['al2_rate_mean']['values'].append(val(5, 'rate_mean'))
        metrics['al2_rate_dual']['values'].append(val(5, 'rate_mean_dual'))
        metrics['al2_power_mean']['values'].append(val(5, 'power_mean'))
        metrics['al2_shutter_duration']['values'].append(val(5, 'shutter_duration'))

        # Junction oxidation (layer 4)
        metrics['jox_pressure']['values'].append(val(4, 'pressure_actual'))
        metrics['jox_duration']['values'].append(val(4, 'duration'))

        # Capping oxidation (layer 6)
        metrics['cap_pressure']['values'].append(val(6, 'pressure_actual'))
        metrics['cap_duration']['values'].append(val(6, 'duration'))

        # Ar etch (layer 1)
        metrics['etch_beam_v']['values'].append(val(1, 'beam_voltage'))
        metrics['etch_beam_i']['values'].append(val(1, 'beam_current'))
        metrics['etch_ar_flow']['values'].append(val(1, 'ar_flow'))

        # Oxidation peak flow
        metrics['jox_flow']['values'].append(val(4, 'peak_flow'))
        metrics['cap_flow']['values'].append(val(6, 'peak_flow'))

        # Etch time from recipe XML
        d = load_run(run_id)
        etch_t = get_etch_time(d['folder']) if d else None
        metrics['etch_time']['values'].append(etch_t)

        # Base pressure
        metrics['base_pressure']['values'].append(
            summary['run_info']['base_pressure'])

        # MC1 crystal life remaining (last positive value — truncated runs
        # may have trailing 0s from NaN fill)
        if d is not None and len(d['mc1_remaining']) > 0:
            mc1_vals = d['mc1_remaining']
            mc1_pos = mc1_vals[mc1_vals > 0]
            if len(mc1_pos) > 0:
                metrics['mc1_remaining']['values'].append(
                    round(float(mc1_pos[-1]), 1))
            else:
                metrics['mc1_remaining']['values'].append(None)
        else:
            metrics['mc1_remaining']['values'].append(None)

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
    return send_from_directory(APP_DIR, 'index.html')


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


@app.route('/api/comparison')
def api_comparison():
    from flask import request
    run_ids = request.args.get('runs', '')
    run_ids = [r.strip() for r in run_ids.split(',') if r.strip()] or None
    return jsonify(compute_comparison(run_ids))


@app.route('/api/run/<run_id>/valves')
def api_valves(run_id):
    d = load_run(run_id)
    if d is None:
        abort(404)

    t     = d['t']
    layer = d['layer']
    phase = d['phase']

    boundaries = []
    for i in range(1, len(layer)):
        if layer[i] != layer[i - 1] or phase[i] != phase[i - 1]:
            L, P = int(layer[i]), int(phase[i])
            lname = LAYER_NAMES.get(L, f'Layer {L}')
            pname = PHASE_NAMES.get(L, {}).get(P, f'Phase {P}')
            boundaries.append({
                't': round(float(t[i]), 2),
                'layer': L, 'phase': P,
                'label': f'{lname}: {pname}',
            })

    return jsonify({
        't':                 clean(t),
        'layer':             layer.tolist(),
        'phase':             phase.tolist(),
        'substrate_shutter': d['shutter'].tolist(),
        'substrate_cmd':     d['shutter_cmd'].tolist(),
        'ti_source_shutter': d['ti_src_shutter'].tolist(),
        'al_source_shutter': d['al_src_shutter'].tolist(),
        'kdc_ion_shutter':   d['kdc_shutter'].tolist(),
        'mc1_sensor_shutter': d['mc1_shutter'].tolist(),
        'dual_sensor_shutter': d['dual_shutter'].tolist(),
        'uniformity_shutter': clean(d['unif_shutter']),
        'phase_boundaries':  boundaries,
    })


@app.route('/api/config', methods=['GET'])
def api_config():
    return jsonify({'log_dir': LOG_DIR})


@app.route('/api/config', methods=['POST'])
def api_set_config():
    from flask import request
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
        description='Angstrom JJ Log Viewer',
        epilog='''
Expected directory layout:
  <log-dir>/
    CQtm_<RecipeName>-<user>_YYYYMMDD_HHMMSS/
      CQtm_<RecipeName>-<user>_YYYYMMDD_HHMMSS.csv
      CQtm_<RecipeName>-<user>_YYYYMMDD_HHMMSS_details.json
      <RecipeName>_Original_<timestamp>.xml
      <RecipeName>_Complete_<timestamp>.xml
      <RecipeName>_Status_<timestamp>.xml

Each subfolder is one run. The folder name must end with
YYYYMMDD_HHMMSS (the run start timestamp).
''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--log-dir', default=LOG_DIR,
        help='Path to directory containing run folders (default: %(default)s)')
    parser.add_argument(
        '--port', type=int, default=5000,
        help='Port to serve on (default: 5000)')
    args = parser.parse_args()

    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        print(f'Error: log directory does not exist: {LOG_DIR}')
        raise SystemExit(1)

    print('Angstrom JJ Log Viewer')
    print(f'  Log directory : {LOG_DIR}')
    print(f'  Serving       : http://localhost:{args.port}')
    app.run(host='0.0.0.0', port=args.port, debug=False)
