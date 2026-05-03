# test_signal_processing.py
import numpy as np
import pytest
from signal_processing import (
    bandpass_filter_ppg, minmax_norm, preprocess_ppg,
    segment_ppg_beats, check_signal_quality,
    extract_bp_features, extract_glucose_features
)

FS = 125

def make_synthetic_ppg(duration_sec=4, fs=FS, hr_bpm=70, noise=0.02):
    """
    Generate a realistic synthetic PPG waveform.
    Uses a sum of sinusoids to mimic systolic peak shape.
    """
    t       = np.linspace(0, duration_sec, int(duration_sec * fs))
    freq    = hr_bpm / 60.0
    # Fundamental + harmonics to shape the waveform
    ppg  = 0.6 * np.sin(2 * np.pi * freq * t)
    ppg += 0.2 * np.sin(4 * np.pi * freq * t - 0.3)
    ppg += 0.1 * np.sin(6 * np.pi * freq * t - 0.6)
    ppg += noise * np.random.randn(len(t))
    # Normalise to [0, 1] like real MAX30102 output
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())
    return ppg.astype(np.float32)


# ── bandpass_filter_ppg ───────────────────────────────────────────────────────

def test_bandpass_output_shape():
    ppg = make_synthetic_ppg()
    out = bandpass_filter_ppg(ppg)
    assert out.shape == ppg.shape, "Filter must preserve signal length"

def test_bandpass_removes_dc():
    ppg = make_synthetic_ppg() + 5.0    # add large DC offset
    out = bandpass_filter_ppg(ppg)
    assert abs(np.mean(out)) < 0.1, "Bandpass must remove DC offset"


# ── minmax_norm ───────────────────────────────────────────────────────────────

def test_minmax_range():
    ppg = make_synthetic_ppg()
    out = minmax_norm(ppg)
    assert out.min() >= 0.0, "minmax_norm min must be >= 0"
    assert out.max() <= 1.0, "minmax_norm max must be <= 1"

def test_minmax_flat_signal():
    flat = np.ones(125)
    out  = minmax_norm(flat)
    assert not np.any(np.isnan(out)), "Flat signal must not produce NaN"


# ── preprocess_ppg ────────────────────────────────────────────────────────────

def test_preprocess_output_range():
    ppg = make_synthetic_ppg()
    out = preprocess_ppg(ppg)
    assert 0.0 <= out.min() and out.max() <= 1.0

def test_preprocess_no_nan():
    ppg = make_synthetic_ppg()
    out = preprocess_ppg(ppg)
    assert not np.any(np.isnan(out)), "Preprocessed PPG must contain no NaN"


# ── check_signal_quality ──────────────────────────────────────────────────────

def test_quality_good_signal():
    ppg      = make_synthetic_ppg(duration_sec=4)
    ok, msg  = check_signal_quality(ppg)
    assert ok, f"Good signal should pass quality check, got: {msg}"

def test_quality_flat_signal():
    flat     = np.ones(500) * 0.5
    ok, msg  = check_signal_quality(flat)
    assert not ok, "Flat signal should fail quality check"
    assert "amplitude" in msg.lower()

def test_quality_clipped_signal():
    clipped  = np.ones(500)    # all at maximum
    ok, msg  = check_signal_quality(clipped)
    assert not ok

def test_quality_short_signal():
    ppg = make_synthetic_ppg(duration_sec=0.5)  # too short for peaks
    ok, _ = check_signal_quality(ppg)
    # May or may not pass depending on length — just check no crash
    assert isinstance(ok, bool)


# ── segment_ppg_beats ─────────────────────────────────────────────────────────

def test_segment_returns_list():
    ppg      = preprocess_ppg(make_synthetic_ppg(duration_sec=4))
    segments = segment_ppg_beats(ppg)
    assert isinstance(segments, list), "Should return a list"
    assert len(segments) > 0, "Should find at least one beat in 4 seconds"

def test_segment_correct_length():
    ppg      = preprocess_ppg(make_synthetic_ppg(duration_sec=4))
    segments = segment_ppg_beats(ppg, window_sec=1.0)
    for seg in segments:
        assert len(seg) == FS, f"Each segment must be {FS} samples, got {len(seg)}"

def test_segment_no_nan():
    ppg      = preprocess_ppg(make_synthetic_ppg(duration_sec=4))
    segments = segment_ppg_beats(ppg)
    for seg in segments:
        assert not np.any(np.isnan(seg)), "Segments must not contain NaN"


# ── extract_bp_features ───────────────────────────────────────────────────────

def test_bp_features_returns_dict():
    ppg      = preprocess_ppg(make_synthetic_ppg(duration_sec=4))
    segments = segment_ppg_beats(ppg)
    feats    = extract_bp_features(segments[0])
    assert isinstance(feats, dict), "Should return a dict"
    assert feats is not None, "Should not return None for valid segment"

def test_bp_features_expected_keys():
    ppg      = preprocess_ppg(make_synthetic_ppg(duration_sec=4))
    segments = segment_ppg_beats(ppg)
    feats    = extract_bp_features(segments[0])
    expected = ['HR', 'IH', 'IL', 'PIR', 'Meu', 'ppg_std',
                'rise_time', 'pulse_width', 'apg_a', 'apg_b',
                'aging_index', 'reflection_index', 'auc',
                'max_slope', 'ppg_skew', 'ppg_kurt']
    for key in expected:
        assert key in feats, f"Missing feature: {key}"

def test_bp_features_no_nan():
    ppg      = preprocess_ppg(make_synthetic_ppg(duration_sec=4))
    segments = segment_ppg_beats(ppg)
    feats    = extract_bp_features(segments[0])
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} is NaN"

def test_bp_features_hr_range():
    ppg      = preprocess_ppg(make_synthetic_ppg(duration_sec=4, hr_bpm=70))
    segments = segment_ppg_beats(ppg)
    feats    = extract_bp_features(segments[0])
    assert 40 < feats['HR'] < 200, f"HR out of range: {feats['HR']}"

def test_bp_features_flat_returns_none():
    flat  = np.ones(125) * 0.5
    feats = extract_bp_features(flat)
    assert feats is None, "Flat segment should return None"


# ── extract_glucose_features ──────────────────────────────────────────────────

def test_glucose_features_length():
    ppg   = preprocess_ppg(make_synthetic_ppg(duration_sec=4))
    segs  = segment_ppg_beats(ppg)
    feats = extract_glucose_features(segs[0])
    assert len(feats) == 20, f"Expected 20 glucose features, got {len(feats)}"

def test_glucose_features_no_nan():
    ppg   = preprocess_ppg(make_synthetic_ppg(duration_sec=4))
    segs  = segment_ppg_beats(ppg)
    feats = extract_glucose_features(segs[0])
    assert not np.any(np.isnan(feats)), "Glucose features must not contain NaN"


# ── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    pytest.main([__file__, '-v'])