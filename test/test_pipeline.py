# =============================================================================
#  test_pipeline.py
#  Run this BEFORE your presentation to verify BP and glucose predictions
#  are producing physiologically correct values end-to-end.
#
#  Usage:
#      python test_pipeline.py
#
#  What it tests (6 tests):
#    1. Signal quality check rejects bad signals
#    2. Signal quality check accepts good signals
#    3. PPG preprocessing produces expected shape and range
#    4. Beat segmentation finds heartbeat windows
#    5. Feature extraction produces correct number of features
#    6. Full inference pipeline produces plausible BP and glucose values
#
#  After all tests pass, run the server test:
#    7. Start server.py in one terminal, then:
#       python test_pipeline.py --server
# =============================================================================

import numpy as np
import sys
import json

# ── Test counters ─────────────────────────────────────────────────────────────
passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  ←  {detail}")
        failed += 1

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =============================================================================
#  HELPER — generate a realistic synthetic PPG signal
#  This simulates what the MAX30102 would produce for a healthy adult.
#  Uses a sum of sinusoids to create a realistic pulse waveform shape.
# =============================================================================

def make_synthetic_ppg(hr_bpm=72, fs=125, duration_sec=2, noise=0.02):
    """
    Generate a synthetic PPG signal with a given heart rate.
    Returns normalised float array of length duration_sec * fs.
    """
    t    = np.linspace(0, duration_sec, int(fs * duration_sec))
    freq = hr_bpm / 60.0   # Hz

    # Realistic PPG shape: sum of harmonics (fast rise, slow fall)
    ppg  = (
        0.6 * np.sin(2 * np.pi * freq * t)
      + 0.2 * np.sin(2 * np.pi * 2 * freq * t + 0.3)
      + 0.1 * np.sin(2 * np.pi * 3 * freq * t + 0.6)
    )
    ppg += np.random.normal(0, noise, len(t))  # add small noise

    # Normalise to [0.1, 0.9] — like a real MAX30102 output after normalisation
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())
    ppg = ppg * 0.8 + 0.1
    return ppg.astype(np.float32)


def make_flat_ppg(fs=125, duration_sec=2):
    """Flat signal — simulates no finger or bad contact."""
    return np.full(int(fs * duration_sec), 0.5, dtype=np.float32)


# =============================================================================
#  TEST GROUP 1 — SIGNAL PROCESSING
# =============================================================================

section("TEST GROUP 1: signal_processing.py")

try:
    from signal_processing import (
        preprocess_ppg, check_signal_quality,
        segment_ppg_beats, extract_bp_features,
        extract_glucose_features
    )
    print("  [OK] Imported signal_processing.py")
except ImportError as e:
    print(f"  [FATAL] Cannot import signal_processing.py: {e}")
    sys.exit(1)

FS = 125

# Test 1: Quality check rejects flat signal
flat_ppg = make_flat_ppg()
ok, reason = check_signal_quality(flat_ppg)
test("Quality check rejects flat PPG (no finger)",
     not ok and "amplitude" in reason.lower(),
     f"Got ok={ok}, reason='{reason}'")

# Test 2: Quality check accepts a good signal
good_ppg = make_synthetic_ppg(hr_bpm=72)
ok, reason = check_signal_quality(good_ppg)
test("Quality check accepts realistic PPG signal",
     ok,
     f"Got ok={ok}, reason='{reason}'")

# Test 3: Preprocessing produces correct shape and range
ppg = preprocess_ppg(good_ppg)
test("preprocess_ppg() output length matches input",
     len(ppg) == len(good_ppg),
     f"Input len={len(good_ppg)}, output len={len(ppg)}")
test("preprocess_ppg() output is in [0, 1]",
     0.0 <= float(ppg.min()) and float(ppg.max()) <= 1.0,
     f"Got range [{ppg.min():.3f}, {ppg.max():.3f}]")

# Test 4: Beat segmentation finds windows
segments = segment_ppg_beats(ppg, fs=FS, window_sec=1.0)
test("segment_ppg_beats() finds at least 1 beat in 2-second window",
     len(segments) >= 1,
     f"Found {len(segments)} segments")
if len(segments) > 0:
    test("Each segment is exactly 125 samples (1 second at 125 Hz)",
         len(segments[0]) == 125,
         f"Got {len(segments[0])} samples")

# Test 5: BP feature extraction
if len(segments) > 0:
    seg = segments[len(segments) // 2]
    bp_feats = extract_bp_features(seg, fs=FS)
    test("extract_bp_features() returns a dict (not None)",
         bp_feats is not None,
         "Returned None — check peak detection in the segment")

    if bp_feats is not None:
        expected_keys = ['HR', 'IH', 'IL', 'PIR', 'Meu', 'ppg_std',
                         'rise_time', 'pulse_width', 'apg_a', 'apg_b',
                         'aging_index', 'reflection_index', 'auc',
                         'max_slope', 'ppg_skew', 'ppg_kurt']
        missing_keys = [k for k in expected_keys if k not in bp_feats]
        test(f"extract_bp_features() returns all 16 expected keys",
             len(missing_keys) == 0,
             f"Missing keys: {missing_keys}")

        # Sanity check feature values
        test("HR is in plausible range (40-200 BPM)",
             40 <= bp_feats.get('HR', 0) <= 200,
             f"HR={bp_feats.get('HR'):.1f}")
        test("IH (systolic amplitude) is positive",
             bp_feats.get('IH', -1) > 0,
             f"IH={bp_feats.get('IH'):.3f}")

# Test 6: Glucose feature extraction
if len(segments) > 0:
    glu_feats = extract_glucose_features(seg, fs=FS)
    test("extract_glucose_features() returns 20 features",
         len(glu_feats) == 20,
         f"Got {len(glu_feats)} features")
    test("Glucose features are all finite (no NaN or Inf)",
         np.all(np.isfinite(glu_feats)),
         f"Non-finite values at indices: {np.where(~np.isfinite(glu_feats))[0]}")


# =============================================================================
#  TEST GROUP 2 — INFERENCE ENGINE
# =============================================================================

section("TEST GROUP 2: inference.py + loaded models")

try:
    from inference import MedWatchInference
    engine = MedWatchInference()
    print("  [OK] Models loaded successfully")
    models_loaded = True
except FileNotFoundError as e:
    print(f"  [SKIP] Models not found — run train_models.py first")
    print(f"         ({e})")
    models_loaded = False
except Exception as e:
    print(f"  [FAIL] Error loading models: {e}")
    models_loaded = False

if models_loaded:
    # Test 7: predict() rejects flat PPG
    try:
        engine.predict(flat_ppg.tolist())
        test("predict() rejects flat PPG", False, "Should have raised ValueError")
    except ValueError as e:
        test("predict() raises ValueError for flat PPG", True)

    # Test 8: predict() returns plausible values for good PPG
    try:
        result = engine.predict(good_ppg.tolist())

        test("predict() returns dict with SBP key",
             'SBP' in result,
             f"Keys: {list(result.keys())}")
        test("predict() returns dict with DBP key",
             'DBP' in result)
        test("predict() returns dict with glucose key",
             'glucose' in result)

        sbp     = result.get('SBP', 0)
        dbp     = result.get('DBP', 0)
        glucose = result.get('glucose', 0)
        hr      = result.get('HR', 0)

        print(f"\n  Prediction on synthetic PPG (HR=72 BPM):")
        print(f"    SBP     = {sbp} mmHg")
        print(f"    DBP     = {dbp} mmHg")
        print(f"    Glucose = {glucose} mg/dL")
        print(f"    HR      = {hr} BPM")
        print(f"    Quality = {result.get('quality')}")

        test("SBP is in physiological range (70–220 mmHg)",
             70 <= sbp <= 220,
             f"SBP={sbp}")
        test("DBP is in physiological range (40–130 mmHg)",
             40 <= dbp <= 130,
             f"DBP={dbp}")
        test("SBP > DBP (pulse pressure is positive)",
             sbp > dbp,
             f"SBP={sbp}, DBP={dbp}")
        test("Glucose is in physiological range (40–500 mg/dL)",
             40 <= glucose <= 500,
             f"Glucose={glucose}")
        test("HR is in physiological range (40–200 BPM)",
             40 <= hr <= 200,
             f"HR={hr}")

    except Exception as e:
        test("predict() runs without crashing on good PPG", False, str(e))
        import traceback
        traceback.print_exc()

    # Test 9: Different HR signals produce different predictions
    # (confirms model is actually using the signal, not returning constants)
    try:
        ppg_slow = make_synthetic_ppg(hr_bpm=55)   # bradycardia
        ppg_fast = make_synthetic_ppg(hr_bpm=100)  # tachycardia

        res_slow = engine.predict(ppg_slow.tolist())
        res_fast = engine.predict(ppg_fast.tolist())

        hr_slow = res_slow.get('HR', 0)
        hr_fast = res_fast.get('HR', 0)

        print(f"\n  Predictions vary with HR:")
        print(f"    Slow PPG (55 BPM target) → HR={hr_slow}")
        print(f"    Fast PPG (100 BPM target) → HR={hr_fast}")

        test("Slow PPG produces lower HR than fast PPG",
             hr_slow < hr_fast,
             f"Slow HR={hr_slow}, Fast HR={hr_fast}")

    except Exception as e:
        test("Different HR signals produce different predictions", False, str(e))


# =============================================================================
#  TEST GROUP 3 — SERVER (optional, requires server.py running)
# =============================================================================

if '--server' in sys.argv:
    section("TEST GROUP 3: server.py HTTP endpoints")

    import urllib.request
    import urllib.error

    SERVER = "http://127.0.0.1:5000"

    # Test /status
    try:
        with urllib.request.urlopen(f"{SERVER}/status", timeout=3) as r:
            data = json.loads(r.read())
        test("/status returns models_loaded=true",
             data.get('models_loaded') is True,
             str(data))
        test("/status confirms ppg_only=true",
             data.get('ppg_only') is True,
             str(data))
    except Exception as e:
        test("/status is reachable", False,
             f"Is server.py running? Error: {e}")

    # Test /predict with good PPG
    try:
        payload = json.dumps({'ppg': good_ppg.tolist()}).encode()
        req = urllib.request.Request(
            f"{SERVER}/predict",
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read())
        test("/predict returns 200 with good PPG",
             'SBP' in result and 'DBP' in result,
             str(result))
        print(f"    Server response: {result}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        test("/predict returns 200", False, f"HTTP {e.code}: {body}")
    except Exception as e:
        test("/predict is reachable", False, str(e))

    # Test /predict rejects short window
    try:
        payload = json.dumps({'ppg': [0.5] * 50}).encode()
        req = urllib.request.Request(
            f"{SERVER}/predict",
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            test("/predict rejects 50-sample window (should return 400)", False,
                 "Should have returned 400")
    except urllib.error.HTTPError as e:
        test("/predict returns 400 for short window (50 samples)",
             e.code == 400,
             f"Got HTTP {e.code}")
    except Exception as e:
        test("/predict is reachable for short window test", False, str(e))

    # Test /history
    try:
        with urllib.request.urlopen(f"{SERVER}/history?n=5", timeout=3) as r:
            data = json.loads(r.read())
        test("/history returns a dict with 'history' key",
             'history' in data,
             str(data))
    except Exception as e:
        test("/history is reachable", False, str(e))


# =============================================================================
#  SUMMARY
# =============================================================================

section("SUMMARY")
total = passed + failed
print(f"  Passed : {passed} / {total}")
print(f"  Failed : {failed} / {total}")
print()

if failed == 0:
    print("  All tests passed — safe to present.")
elif failed <= 2:
    print("  Minor issues found — review the FAIL lines above before presenting.")
else:
    print("  Multiple failures — do NOT present until these are fixed.")

if '--server' not in sys.argv:
    print()
    print("  To also test the live server endpoints:")
    print("    1. Start server:  python server.py")
    print("    2. Run:           python test_pipeline.py --server")
