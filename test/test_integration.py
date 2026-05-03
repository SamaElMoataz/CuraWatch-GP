# test_integration.py
import numpy as np
from signal_processing import preprocess_ppg, segment_ppg_beats, \
                               check_signal_quality, extract_bp_features, \
                               extract_glucose_features

FS = 125

def make_synthetic_ppg(duration_sec=4, fs=FS, hr_bpm=70):
    t    = np.linspace(0, duration_sec, int(duration_sec * fs))
    freq = hr_bpm / 60.0
    ppg  = 0.6 * np.sin(2 * np.pi * freq * t)
    ppg += 0.2 * np.sin(4 * np.pi * freq * t - 0.3)
    ppg += 0.1 * np.sin(6 * np.pi * freq * t - 0.6)
    ppg += 0.02 * np.random.randn(len(t))
    ppg  = (ppg - ppg.min()) / (ppg.max() - ppg.min())
    return ppg.astype(np.float32)

def test_full_pipeline():
    print("\n── Full pipeline integration test ──")

    # Step 1 — simulate raw ESP32 input
    raw_ppg = make_synthetic_ppg(duration_sec=4)
    print(f"  Raw PPG: {len(raw_ppg)} samples, range [{raw_ppg.min():.3f}, {raw_ppg.max():.3f}]")

    # Step 2 — quality check
    ok, reason = check_signal_quality(raw_ppg)
    print(f"  Quality check: {'PASS' if ok else 'FAIL'} — {reason}")
    assert ok

    # Step 3 — preprocess
    ppg = preprocess_ppg(raw_ppg)
    print(f"  After preprocess: range [{ppg.min():.3f}, {ppg.max():.3f}]")

    # Step 4 — segment
    segments = segment_ppg_beats(ppg)
    print(f"  Segments found: {len(segments)}, each {len(segments[0])} samples")
    assert len(segments) > 0

    # Step 5 — extract BP features
    ppg_seg = segments[len(segments) // 2]
    bp_feats = extract_bp_features(ppg_seg)
    print(f"  BP features: {list(bp_feats.keys())}")
    print(f"  HR={bp_feats['HR']:.1f}  IH={bp_feats['IH']:.3f}  "
          f"aging_index={bp_feats['aging_index']:.4f}")
    assert bp_feats is not None

    # Step 6 — extract glucose features
    glu_feats = extract_glucose_features(ppg_seg)
    print(f"  Glucose features: {len(glu_feats)} values, "
          f"no NaN: {not np.any(np.isnan(glu_feats))}")
    assert len(glu_feats) == 20

    print("\n  ✓ Full pipeline passed\n")

if __name__ == '__main__':
    test_full_pipeline()