# =============================================================================
#  server.py  —  PPG-only + PPG+ECG dual-mode version
#
#  Run with:  python server.py
#  Default:   http://0.0.0.0:5000
# =============================================================================

from flask import Flask, request, jsonify
import numpy as np
import traceback
import time
import os

from inference import MedWatchInference

app = Flask(__name__)

# ── Load inference engine once at startup ─────────────────────────────────────
print("[Server] Loading models...")
try:
    engine = MedWatchInference()
    print("[Server] Models loaded — ready")
except FileNotFoundError as e:
    print(f"[Server] ERROR: {e}")
    print("[Server] Run train_models.py first!")
    engine = None


# =============================================================================
#  /predict  — main endpoint called by ESP32
# =============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives raw PPG (and optional ECG) window from ESP32.

    Request body:
    {
        "ppg": [0.51, 0.53, ...],   ← minimum 125 float values (1 second at 125 Hz)
        "ecg": [0.01, -0.02, ...]   ← optional, same length; enables ECG-assisted BP
    }

    Response body:
    {
        "SBP":          120.5,      ← systolic blood pressure (mmHg)
        "DBP":          78.2,       ← diastolic blood pressure (mmHg)
        "glucose":      95.0,       ← blood glucose (mg/dL)
        "HR":           72.0,       ← heart rate (BPM)
        "mode":         "PPG+ECG",  ← "PPG-only" or "PPG+ECG"
        "calibrated":   false,
        "quality":      "good",
        "timestamp":    1710000000,
        "inference_ms": 45.2
    }
    """
    if engine is None:
        return jsonify({
            'error': 'Models not loaded — run train_models.py first'
        }), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid JSON body'}), 400

    ppg = data.get('ppg')
    ecg = data.get('ecg')   # optional

    if ppg is None:
        return jsonify({'error': 'Missing ppg array in request'}), 400

    if len(ppg) < 125:
        return jsonify({
            'error': f'PPG window too short: {len(ppg)} samples (minimum 125)'
        }), 400

    if ecg is not None and len(ecg) < 125:
        # ECG too short to be useful — ignore it, run PPG-only
        print(f"[Predict] ECG array too short ({len(ecg)} samples) — ignored")
        ecg = None

    t_start = time.time()
    try:
        result = engine.predict(ppg, raw_ecg=ecg)
        result['timestamp']    = int(time.time())
        result['inference_ms'] = round((time.time() - t_start) * 1000, 1)

        print(f"[Predict] SBP={result['SBP']}  DBP={result['DBP']}  "
              f"Glu={result['glucose']}  HR={result['HR']}  "
              f"Mode={result['mode']}  ({result['inference_ms']}ms)")

        return jsonify(result), 200

    except ValueError as e:
        print(f"[Predict] Signal rejected: {e}")
        return jsonify({'error': str(e), 'quality': 'poor'}), 400

    except Exception as e:
        print(f"[Predict] Unexpected error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal inference error'}), 500


# =============================================================================
#  /calibrate  — personalise predictions with cuff reference readings
# =============================================================================

@app.route('/calibrate', methods=['POST'])
def calibrate():
    """
    Provide reference measurements from a BP cuff and glucometer.
    Collect 3–5 paired readings for best results.
    After calibration, errors typically reduce by 30–50%.

    Request body:
    {
        "pred_sbp": [118, 122, 115],
        "true_sbp": [122, 126, 118],
        "pred_dbp": [76,  79,  74 ],
        "true_dbp": [78,  82,  77 ],
        "pred_glu": [90, 105, 88  ],   ← optional
        "true_glu": [95, 110, 92  ]    ← optional
    }
    """
    if engine is None:
        return jsonify({'error': 'Models not loaded'}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    for key in ['pred_sbp', 'true_sbp', 'pred_dbp', 'true_dbp']:
        if key not in data:
            return jsonify({'error': f'Missing field: {key}'}), 400

    try:
        engine.calibrate(
            pred_sbp=data['pred_sbp'], true_sbp=data['true_sbp'],
            pred_dbp=data['pred_dbp'], true_dbp=data['true_dbp'],
            pred_glu=data.get('pred_glu'), true_glu=data.get('true_glu')
        )
        return jsonify({
            'status':    'calibrated',
            'n_samples': len(data['pred_sbp'])
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
#  /calibrate/reset
# =============================================================================

@app.route('/calibrate/reset', methods=['POST'])
def reset_calibration():
    """Reset calibration to factory defaults."""
    if engine is None:
        return jsonify({'error': 'Models not loaded'}), 503

    engine.calibrator.__init__()

    cal_path = "models/calibration.pkl"
    if os.path.exists(cal_path):
        os.remove(cal_path)

    return jsonify({'status': 'calibration reset to defaults'}), 200


# =============================================================================
#  /status
# =============================================================================

@app.route('/status', methods=['GET'])
def status():
    """Health check — confirms server is running and which models are loaded."""
    return jsonify({
        'status':            'running',
        'models_loaded':     engine is not None,
        'ecg_model_loaded':  engine.ecg_model_available if engine else False,
        'calibrated':        engine.calibrator.calibrated if engine else False,
        'mode_available':    ('PPG+ECG' if (engine and engine.ecg_model_available)
                              else 'PPG-only'),
    }), 200


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[Server] Starting on http://0.0.0.0:{port}")
    print("[Server] Send 'ppg' only for PPG-only mode")
    print("[Server] Send 'ppg' + 'ecg' for ECG-assisted mode (if model loaded)")
    print("[Server] Make sure ESP32 and server are on the same WiFi network")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
