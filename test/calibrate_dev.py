# calibrate_dev.py
# Run this while server.py is running in another terminal.
# Fill in your actual measured values before running.

import requests
import json

BASE = "http://localhost:5000"

# ─────────────────────────────────────────────────────────────────────────────
#  FILL IN YOUR VALUES HERE
#  Each list must have the same number of entries.
#  Minimum 2 readings required, 3-5 recommended.
# ─────────────────────────────────────────────────────────────────────────────

# Blood pressure — watch predictions (from /predict responses)
PREDICTED_SBP = [118, 122, 115, 120, 117]
PREDICTED_DBP = [75,  78,  73,  77,  74]

# Blood pressure — real cuff readings
TRUE_SBP = [122, 126, 119, 124, 121]
TRUE_DBP = [78,  82,  76,  80,  77]

# Glucose — watch predictions (from /predict responses)
PREDICTED_GLUCOSE = [92, 95, 89, 93, 91]

# Glucose — real glucometer readings
TRUE_GLUCOSE = [98, 101, 94, 99, 96]

# ─────────────────────────────────────────────────────────────────────────────


def check_status():
    """Verify server is running and check current calibration state."""
    print("\n── Checking server status ──")
    r = requests.get(f"{BASE}/status")
    data = r.json()
    print(f"   Server running  : {data['status']}")
    print(f"   Models loaded   : {data['models_loaded']}")
    print(f"   Already calibrated: {data['calibrated']}")
    print(f"   Total predictions so far: {data['n_predictions']}")
    return data['models_loaded']


def send_calibration():
    """Send calibration data to server."""
    print("\n── Sending calibration data ──")

    payload = {
        "pred_sbp": PREDICTED_SBP,
        "true_sbp": TRUE_SBP,
        "pred_dbp": PREDICTED_DBP,
        "true_dbp": TRUE_DBP,
        "pred_glu": PREDICTED_GLUCOSE,
        "true_glu": TRUE_GLUCOSE,
    }

    # Show what we are sending
    print(f"   Number of readings: {len(PREDICTED_SBP)}")
    print(f"\n   SBP errors before calibration:")
    for i, (p, t) in enumerate(zip(PREDICTED_SBP, TRUE_SBP)):
        print(f"     Reading {i+1}: predicted={p}  true={t}  error={t-p:+d} mmHg")

    print(f"\n   DBP errors before calibration:")
    for i, (p, t) in enumerate(zip(PREDICTED_DBP, TRUE_DBP)):
        print(f"     Reading {i+1}: predicted={p}  true={t}  error={t-p:+d} mmHg")

    print(f"\n   Glucose errors before calibration:")
    for i, (p, t) in enumerate(zip(PREDICTED_GLUCOSE, TRUE_GLUCOSE)):
        print(f"     Reading {i+1}: predicted={p}  true={t}  error={t-p:+d} mg/dL")

    # Send to server
    r = requests.post(f"{BASE}/calibrate", json=payload)
    print(f"\n   Response status : {r.status_code}")
    print(f"   Response body   : {json.dumps(r.json(), indent=4)}")
    return r.status_code == 200


def verify_calibration():
    """
    Check status again to confirm calibration was saved,
    then show what the correction factors are.
    """
    print("\n── Verifying calibration was saved ──")
    r = requests.get(f"{BASE}/status")
    data = r.json()
    print(f"   Calibrated: {data['calibrated']}")

    if data['calibrated']:
        print("   ✓ Calibration saved successfully")
    else:
        print("   ✗ Calibration does not appear to have been saved")


def test_prediction_after_calibration():
    """
    Send a synthetic PPG to /predict and show that
    'calibrated: true' appears in the response.
    """
    import numpy as np

    print("\n── Testing prediction after calibration ──")

    # Generate synthetic PPG
    fs  = 125
    t   = np.linspace(0, 4, 4 * fs)
    hr  = 70 / 60.0
    ppg = 0.6 * np.sin(2 * np.pi * hr * t)
    ppg += 0.2 * np.sin(4 * np.pi * hr * t - 0.3)
    ppg += 0.1 * np.sin(6 * np.pi * hr * t - 0.6)
    ppg += 0.02 * np.random.randn(len(t))
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())

    r = requests.post(f"{BASE}/predict", json={"ppg": ppg.tolist()})

    if r.status_code == 200:
        result = r.json()
        print(f"   SBP       : {result['SBP']} mmHg")
        print(f"   DBP       : {result['DBP']} mmHg")
        print(f"   Glucose   : {result['glucose']} mg/dL")
        print(f"   Calibrated: {result['calibrated']}")
        if result['calibrated']:
            print("   ✓ Predictions are now using personal calibration")
        else:
            print("   ✗ Calibration flag is False — something went wrong")
    else:
        print(f"   Prediction failed: {r.json()}")


def show_expected_improvement():
    """
    Calculate and display what the errors looked like before
    and what they should look like after calibration.
    This uses a simple linear correction estimate.
    """
    import numpy as np

    print("\n── Expected error improvement ──")

    sbp_errors_before = [abs(t - p) for t, p in zip(TRUE_SBP, PREDICTED_SBP)]
    dbp_errors_before = [abs(t - p) for t, p in zip(TRUE_DBP, PREDICTED_DBP)]
    glu_errors_before = [abs(t - p) for t, p in zip(TRUE_GLUCOSE, PREDICTED_GLUCOSE)]

    print(f"   Before calibration:")
    print(f"     SBP MAE : {np.mean(sbp_errors_before):.1f} mmHg")
    print(f"     DBP MAE : {np.mean(dbp_errors_before):.1f} mmHg")
    print(f"     Glu MAE : {np.mean(glu_errors_before):.1f} mg/dL")

    # After calibration the residual error comes from
    # biological variability, not systematic offset.
    # Typical improvement is 30-50% reduction in MAE.
    print(f"\n   Expected after calibration (estimate):")
    print(f"     SBP MAE : ~{np.mean(sbp_errors_before)*0.5:.1f} - "
          f"{np.mean(sbp_errors_before)*0.7:.1f} mmHg")
    print(f"     DBP MAE : ~{np.mean(dbp_errors_before)*0.5:.1f} - "
          f"{np.mean(dbp_errors_before)*0.7:.1f} mmHg")
    print(f"     Glu MAE : ~{np.mean(glu_errors_before)*0.5:.1f} - "
          f"{np.mean(glu_errors_before)*0.7:.1f} mg/dL")
    print(f"\n   Note: actual improvement depends on how consistent")
    print(f"   the systematic error is across readings.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  MedWatch Development Calibration Tool")
    print("=" * 55)

    # Step 1 — verify server is up
    if not check_status():
        print("\n✗ Models not loaded — run train_models.py first")
        exit(1)

    # Step 2 — show expected improvement estimate
    show_expected_improvement()

    # Step 3 — send calibration
    success = send_calibration()
    if not success:
        print("\n✗ Calibration failed — check server terminal for errors")
        exit(1)

    # Step 4 — verify it was saved
    verify_calibration()

    # Step 5 — test a prediction to confirm calibration is active
    test_prediction_after_calibration()

    print("\n" + "=" * 55)
    print("  Calibration complete.")
    print("  All future predictions will use your personal correction.")
    print("=" * 55)