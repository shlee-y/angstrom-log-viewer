"""
Unified Flask backend for Angstrom log viewers.
Serves both JJ evaporator and Ta sputterer data under
/api/<machine>/... URL prefixes from a single server.
"""

import os
import sys
import importlib.util

from flask import Flask, jsonify, request, send_from_directory, abort
from flask.json.provider import DefaultJSONProvider
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_JJ_DIR = os.path.join(APP_DIR, 'Angstrom log')
DEFAULT_TA_DIR = os.path.join(APP_DIR, 'Ta log')

# ---------------------------------------------------------------------------
# Load existing server modules (their functions, not their Flask apps)
# ---------------------------------------------------------------------------

def _load_module(name, filename):
    """Import a .py file as a module without running its __main__ block."""
    path = os.path.join(APP_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

jj_mod = _load_module('jj_mod', 'server.py')
ta_mod = _load_module('ta_mod', 'ta-server.py')

# ---------------------------------------------------------------------------
# Machine registry
# ---------------------------------------------------------------------------

machines = {
    'jj': {
        'mod':      jj_mod,
        'name':     'JJ Evaporator',
        'subtitle': 'Al/AlOx/Al Josephson junction',
        'log_dir':  DEFAULT_JJ_DIR,
        'accent':   '#4fd1c5',
    },
    'ta': {
        'mod':      ta_mod,
        'name':     'Ta Sputterer',
        'subtitle': 'Tantalum pulsed DC sputtering',
        'log_dir':  DEFAULT_TA_DIR,
        'accent':   '#f6ad55',
    },
}

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_machine(machine_id):
    """Look up a machine by id, set its LOG_DIR, and return its config dict.
    Aborts with 404 if the machine id is unknown.
    """
    m = machines.get(machine_id)
    if m is None:
        abort(404, description=f'Unknown machine: {machine_id}')
    # Ensure the module's LOG_DIR global is in sync
    m['mod'].LOG_DIR = m['log_dir']
    return m


def _jj_valves(run_id):
    """Build the valves response dict for the JJ evaporator."""
    mod = machines['jj']['mod']
    mod.LOG_DIR = machines['jj']['log_dir']
    d = mod.load_run(run_id)
    if d is None:
        return None

    t     = d['t']
    layer = d['layer']
    phase = d['phase']

    boundaries = []
    for i in range(1, len(layer)):
        if layer[i] != layer[i - 1] or phase[i] != phase[i - 1]:
            L, P = int(layer[i]), int(phase[i])
            lname = mod.LAYER_NAMES.get(L, f'Layer {L}')
            pname = mod.PHASE_NAMES.get(L, {}).get(P, f'Phase {P}')
            boundaries.append({
                't': round(float(t[i]), 2),
                'layer': L, 'phase': P,
                'label': f'{lname}: {pname}',
            })

    return {
        't':                   mod.clean(t),
        'layer':               layer.tolist(),
        'phase':               phase.tolist(),
        'substrate_shutter':   d['shutter'].tolist(),
        'substrate_cmd':       d['shutter_cmd'].tolist(),
        'ti_source_shutter':   d['ti_src_shutter'].tolist(),
        'al_source_shutter':   d['al_src_shutter'].tolist(),
        'kdc_ion_shutter':     d['kdc_shutter'].tolist(),
        'mc1_sensor_shutter':  d['mc1_shutter'].tolist(),
        'dual_sensor_shutter': d['dual_shutter'].tolist(),
        'uniformity_shutter':  mod.clean(d['unif_shutter']),
        'phase_boundaries':    boundaries,
    }


def _ta_valves(run_id):
    """Build the valves response dict for the Ta sputterer."""
    mod = machines['ta']['mod']
    mod.LOG_DIR = machines['ta']['log_dir']
    d = mod.load_run(run_id)
    if d is None:
        return None

    t     = d['t']
    layer = d['layer']
    phase = d['phase']

    phase_names = mod.get_phase_names(run_id)

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

    return {
        't':                 mod.clean(t),
        'layer':             layer.tolist(),
        'phase':             phase.tolist(),
        'substrate_shutter': d['substrate_shutter'].tolist(),
        'substrate_cmd':     d['substrate_cmd'].tolist(),
        'source_shutter':    d['src_shutter'].tolist(),
        'sensor_shutter':    d['sensor_shutter'].tolist(),
        'phase_boundaries':  boundaries,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/')
def index():
    return send_from_directory(APP_DIR, 'app.html')


@app.route('/api/machines')
def api_machines():
    return jsonify([
        {
            'id':       mid,
            'name':     m['name'],
            'subtitle': m['subtitle'],
            'log_dir':  m['log_dir'],
            'accent':   m['accent'],
        }
        for mid, m in machines.items()
    ])


@app.route('/api/<machine_id>/runs')
def api_runs(machine_id):
    m = get_machine(machine_id)
    return jsonify(m['mod'].scan_runs())


@app.route('/api/<machine_id>/run/<run_id>/summary')
def api_summary(machine_id, run_id):
    m = get_machine(machine_id)
    result = m['mod'].compute_summary(run_id)
    if result is None:
        abort(404)
    return jsonify(result)


@app.route('/api/<machine_id>/run/<run_id>/timeseries')
def api_timeseries(machine_id, run_id):
    m = get_machine(machine_id)
    result = m['mod'].get_timeseries(run_id)
    if result is None:
        abort(404)
    return jsonify(result)


@app.route('/api/<machine_id>/run/<run_id>/recipe-diff')
def api_recipe_diff(machine_id, run_id):
    m = get_machine(machine_id)
    result = m['mod'].compute_recipe_diff(run_id)
    if result is None:
        abort(404)
    return jsonify(result)


@app.route('/api/<machine_id>/run/<run_id>/valves')
def api_valves(machine_id, run_id):
    if machine_id == 'jj':
        result = _jj_valves(run_id)
    elif machine_id == 'ta':
        result = _ta_valves(run_id)
    else:
        abort(404, description=f'Unknown machine: {machine_id}')
        return  # unreachable, but keeps linters happy
    if result is None:
        abort(404)
    return jsonify(result)


@app.route('/api/<machine_id>/comparison')
def api_comparison(machine_id):
    m = get_machine(machine_id)
    run_ids = request.args.get('runs', '')
    run_ids = [r.strip() for r in run_ids.split(',') if r.strip()] or None
    return jsonify(m['mod'].compute_comparison(run_ids))


@app.route('/api/<machine_id>/config', methods=['GET'])
def api_get_config(machine_id):
    m = get_machine(machine_id)
    return jsonify({'log_dir': m['log_dir']})


@app.route('/api/<machine_id>/config', methods=['POST'])
def api_set_config(machine_id):
    m = get_machine(machine_id)
    data = request.get_json(force=True)
    new_dir = data.get('log_dir', '').strip()
    if not new_dir or not os.path.isdir(new_dir):
        return jsonify({'error': 'Directory does not exist'}), 400
    m['log_dir'] = new_dir
    m['mod'].LOG_DIR = new_dir
    m['mod']._cache.clear()
    return jsonify({'log_dir': m['log_dir']})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Unified Angstrom Log Viewer (JJ + Ta)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Serves both the JJ evaporator and Ta sputterer log viewers
from a single Flask server.  The frontend picks the machine
from /api/machines and routes all API calls through
/api/<machine>/...

Examples:
  python app.py
  python app.py --jj-dir "D:\\logs\\JJ" --ta-dir "D:\\logs\\Ta" --port 8000
''',
    )
    parser.add_argument(
        '--jj-dir', default=DEFAULT_JJ_DIR,
        help='JJ evaporator log directory (default: %(default)s)')
    parser.add_argument(
        '--ta-dir', default=DEFAULT_TA_DIR,
        help='Ta sputterer log directory (default: %(default)s)')
    parser.add_argument(
        '--port', type=int, default=5000,
        help='Port to serve on (default: 5000)')
    args = parser.parse_args()

    machines['jj']['log_dir'] = args.jj_dir
    machines['ta']['log_dir'] = args.ta_dir

    # Validate directories (warn but don't abort — one machine may be offline)
    for mid, m in machines.items():
        if os.path.isdir(m['log_dir']):
            m['mod'].LOG_DIR = m['log_dir']
        else:
            print(f'  WARNING: {m["name"]} log directory does not exist: {m["log_dir"]}')

    print('Angstrom Unified Log Viewer')
    print(f'  JJ log dir : {machines["jj"]["log_dir"]}')
    print(f'  Ta log dir : {machines["ta"]["log_dir"]}')
    print(f'  Serving    : http://localhost:{args.port}')
    app.run(host='0.0.0.0', port=args.port, debug=False)
