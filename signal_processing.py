# =============================================================================
#  signal_processing.py  —  PPG-only + PPG+ECG dual-mode version
#
#  PPG-only  (MAX30102 wrist PPG):          16 morphology features
#  PPG+ECG   (MAX30102 + AD8232 electrodes): 18 features (adds PTT, WN)
#
#  The ECG path is triggered automatically when the ESP32 sends both signals.
#  If only PPG is received the PPG-only model is used transparently.
# =============================================================================

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, detrend
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

FS = 125


# =============================================================================
#  PPG PREPROCESSING
# =============================================================================

def bandpass_filter_ppg(signal, lowcut=0.5, highcut=8.0, fs=FS, order=4):
    """
    Bandpass filter for PPG signal.
    0.5 Hz removes baseline drift (breathing artefacts).
    8.0 Hz removes high-frequency electrical noise.
    """
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def minmax_norm(signal):
    """Scale signal to [0, 1]."""
    rng = signal.max() - signal.min()
    if rng < 1e-8:
        return signal - signal.min()
    return (signal - signal.min()) / rng


def preprocess_ppg(raw_ppg, fs=FS):
    """
    Full PPG preprocessing pipeline.
    Input:  raw float array from ESP32 (already roughly [0, 1])
    Output: clean, filtered, normalised PPG
    """
    ppg_filtered  = bandpass_filter_ppg(raw_ppg, fs=fs)
    ppg_detrended = detrend(ppg_filtered)
    ppg_norm      = minmax_norm(ppg_detrended)
    return ppg_norm


# =============================================================================
#  ECG PREPROCESSING
# =============================================================================

def preprocess_ecg(raw_ecg, fs=FS):
    """Preprocess ECG signal for R-peak detection."""
    ecg_detrended = detrend(raw_ecg)
    win = int(fs * 0.4) + 1
    if win % 2 == 0:
        win += 1
    win = min(win, len(ecg_detrended) - 1)
    if win < 5:
        return ecg_detrended
    ecg_smooth = savgol_filter(ecg_detrended, window_length=win, polyorder=3)
    peak = np.max(np.abs(ecg_smooth))
    if peak < 1e-8:
        return ecg_smooth
    return ecg_smooth / peak


def detect_r_peaks(ecg, fs=FS):
    """Detect R-peaks from preprocessed ECG."""
    ecg_norm = minmax_norm(ecg)
    peaks, _ = find_peaks(
        ecg_norm,
        height=0.5,
        distance=int(fs * 0.4)
    )
    return peaks


# =============================================================================
#  BEAT SEGMENTATION
# =============================================================================

def segment_ppg_beats(ppg, fs=FS, window_sec=1.0):
    """
    Segment PPG into fixed-length beat windows centred on PPG systolic peaks.
    Returns list of ppg_seg arrays, each of length window_sec * fs.
    """
    ppg_norm = minmax_norm(ppg)
    peaks, _ = find_peaks(ppg_norm, height=0.5, distance=int(fs * 0.4))

    half     = int(window_sec * fs / 2)
    target   = int(window_sec * fs)
    segments = []

    for p in peaks:
        start = p - half
        end   = p + half

        if start < 0 or end > len(ppg):
            continue

        seg = ppg[start:end].copy()
        if len(seg) < target:
            seg = np.pad(seg, (0, target - len(seg)), mode='edge')

        segments.append(seg[:target])

    return segments


def segment_ecg_ppg_beats(ecg, ppg, fs=FS, window_sec=1.0):
    """
    Segment both ECG and PPG into aligned windows centred on ECG R-peaks.
    Both signals must have the same length and sampling rate.

    Returns list of (ecg_seg, ppg_seg) tuples.
    """
    ecg_proc = preprocess_ecg(ecg, fs=fs)
    r_peaks  = detect_r_peaks(ecg_proc, fs=fs)

    half     = int(window_sec * fs / 2)
    target   = int(window_sec * fs)
    segments = []

    n = min(len(ecg), len(ppg))

    for r in r_peaks:
        start = r - half
        end   = r + half

        if start < 0 or end > n:
            continue

        ecg_seg = ecg[start:end].copy()
        ppg_seg = ppg[start:end].copy()

        if len(ecg_seg) < target:
            ecg_seg = np.pad(ecg_seg, (0, target - len(ecg_seg)), mode='edge')
            ppg_seg = np.pad(ppg_seg, (0, target - len(ppg_seg)), mode='edge')

        segments.append((ecg_seg[:target], ppg_seg[:target]))

    return segments


# =============================================================================
#  SIGNAL QUALITY CHECKS
# =============================================================================

def check_ppg_quality(ppg, fs=FS):
    """
    Basic PPG quality checks before running inference.
    Returns (is_ok, reason_string).
    """
    ppg_range = ppg.max() - ppg.min()
    if ppg_range < 0.01:
        return False, "PPG amplitude too low — check sensor placement"

    flat_points = np.sum(np.diff(ppg) == 0)
    if flat_points / len(ppg) > 0.1:
        return False, "PPG clipping detected — reduce LED brightness"

    ppg_norm = minmax_norm(ppg)
    peaks, _ = find_peaks(ppg_norm, height=0.5, distance=int(fs * 0.4))
    if len(peaks) < 2:
        return False, "Could not detect pulse peaks in PPG — check sensor placement"

    hr = 60.0 / np.mean(np.diff(peaks) / fs)
    if hr < 40 or hr > 200:
        return False, f"Heart rate out of range: {hr:.0f} BPM"

    return True, "OK"


def check_ecg_quality(ecg, fs=FS):
    """
    Check if ECG signal is usable.
    Returns (is_ok, reason).
    Called before deciding whether to use PPG+ECG mode.
    """
    if ecg.max() - ecg.min() < 0.05:
        return False, "ECG amplitude too low — check electrode contact"

    r_peaks = detect_r_peaks(preprocess_ecg(ecg, fs=fs), fs=fs)
    if len(r_peaks) < 2:
        return False, "Could not detect R-peaks — poor ECG contact"

    hr = 60.0 / np.mean(np.diff(r_peaks) / fs)
    if hr < 40 or hr > 200:
        return False, f"ECG heart rate out of range: {hr:.0f} BPM"

    return True, "OK"


# =============================================================================
#  FEATURE EXTRACTION — BP (PPG-only, 16 features)
# =============================================================================

def extract_bp_features(ppg_seg, fs=FS):
    """
    Extract 16 PPG-only features for BP prediction from one beat window.

    Feature list (must match FEATURE_COLS_PPG in train_models.py):
      HR, IH, IL, PIR, Meu, ppg_std,
      rise_time, pulse_width,
      apg_a, apg_b, aging_index,
      reflection_index, auc, max_slope,
      ppg_skew, ppg_kurt

    Returns dict, or None if segment is invalid.
    """
    features = {}

    ppg_d1 = np.gradient(ppg_seg)
    ppg_d2 = np.gradient(ppg_d1)

    peaks, _ = find_peaks(ppg_seg, height=np.mean(ppg_seg))
    if len(peaks) == 0:
        peaks, _ = find_peaks(ppg_seg)
    if len(peaks) == 0:
        return None
    sys_peak = peaks[np.argmax(ppg_seg[peaks])]

    troughs, _ = find_peaks(-ppg_seg)
    pre_troughs = troughs[troughs < sys_peak]
    onset_idx   = int(pre_troughs[-1]) if len(pre_troughs) > 0 else 0

    features['HR']      = 60.0 / (len(ppg_seg) / fs)
    features['IH']      = float(ppg_seg[sys_peak])

    post_troughs = troughs[troughs > onset_idx]
    features['IL']      = float(ppg_seg[post_troughs[0]]) if len(post_troughs) > 0 \
                          else float(ppg_seg.min())
    features['PIR']     = features['IH'] / (abs(features['IL']) + 1e-8)
    features['Meu']     = float(np.mean(ppg_seg))
    features['ppg_std'] = float(np.std(ppg_seg))

    features['rise_time'] = (sys_peak - onset_idx) / fs

    half_amp = features['IH'] / 2.0
    above    = np.where(ppg_seg >= half_amp)[0]
    features['pulse_width'] = (above[-1] - above[0]) / fs if len(above) > 1 else 0.0

    search_end = min(sys_peak + 1, len(ppg_d2))
    pos_apg, _ = find_peaks(ppg_d2[:search_end])
    features['apg_a'] = float(ppg_d2[pos_apg[0]]) if len(pos_apg) > 0 \
                        else float(np.max(ppg_d2[:search_end]))

    neg_apg, _ = find_peaks(-ppg_d2[:search_end])
    features['apg_b'] = float(ppg_d2[neg_apg[0]]) if len(neg_apg) > 0 \
                        else float(np.min(ppg_d2[:search_end]))

    features['aging_index']     = features['apg_b'] / (features['apg_a'] + 1e-8)

    total_area = float(np.trapezoid(ppg_seg))
    post_area  = float(np.trapezoid(ppg_seg[sys_peak:]))
    features['reflection_index'] = post_area / (total_area + 1e-8)
    features['auc']              = total_area / fs
    features['max_slope']        = float(np.max(ppg_d1))
    features['ppg_skew']         = float(skew(ppg_seg))
    features['ppg_kurt']         = float(kurtosis(ppg_seg))

    return features


# =============================================================================
#  FEATURE EXTRACTION — BP (PPG+ECG, 18 features)
# =============================================================================

def extract_bp_features_with_ecg(ppg_seg, ecg_seg, fs=FS):
    """
    Extract 18 BP features using both PPG and ECG.
    Adds PTT and WN (Womersley Number) on top of the 16 PPG-only features.
    PTT is the strongest single BP predictor available from a wearable.

    Feature list (must match FEATURE_COLS_ECG in train_models.py):
      PTT, WN,
      HR, IH, IL, PIR, Meu, ppg_std,
      rise_time, pulse_width,
      apg_a, apg_b, aging_index,
      reflection_index, auc, max_slope,
      ppg_skew, ppg_kurt

    Returns dict, or None if PTT cannot be computed.
    """
    features = {}

    # ── ECG R-peak ────────────────────────────────────────────────────────────
    ecg_proc = preprocess_ecg(ecg_seg, fs=fs)
    r_peaks  = detect_r_peaks(ecg_proc, fs=fs)

    if len(r_peaks) == 0:
        r_idx = int(np.argmax(np.abs(ecg_proc)))
    else:
        r_idx = int(r_peaks[np.argmax(ecg_proc[r_peaks])])

    # ── PPG systolic peak after R-peak (PTT window: 50–500 ms) ───────────────
    search_start  = r_idx + int(fs * 0.05)           # ignore first 50 ms
    search_end    = min(r_idx + int(fs * 0.5), len(ppg_seg))
    search_region = ppg_seg[search_start:search_end]

    if len(search_region) < 5:
        return None

    local_peaks, _ = find_peaks(search_region, height=np.mean(search_region))
    if len(local_peaks) == 0:
        sys_peak = search_start + int(np.argmax(search_region))
    else:
        sys_peak = search_start + local_peaks[np.argmax(search_region[local_peaks])]

    # ── PTT ──────────────────────────────────────────────────────────────────
    ptt = (sys_peak - r_idx) / fs
    if ptt <= 0.05 or ptt > 0.5:
        return None

    features['PTT'] = ptt
    # Womersley Number: encodes vessel geometry + blood viscosity + PTT
    # Arterial radius ~3 mm, kinematic viscosity ~3e-6 m²/s
    features['WN']  = np.pi * 0.003 / np.sqrt(3e-6 * (ptt + 1e-8))

    # ── Shared morphology features (same as PPG-only) ─────────────────────────
    ppg_d1 = np.gradient(ppg_seg)
    ppg_d2 = np.gradient(ppg_d1)

    troughs, _ = find_peaks(-ppg_seg)
    pre_troughs = troughs[troughs < sys_peak]
    onset_idx   = int(pre_troughs[-1]) if len(pre_troughs) > 0 else 0

    features['HR']      = 60.0 / (len(ppg_seg) / fs)
    features['IH']      = float(ppg_seg[sys_peak])

    post_troughs = troughs[troughs > onset_idx]
    features['IL']      = float(ppg_seg[post_troughs[0]]) if len(post_troughs) > 0 \
                          else float(ppg_seg.min())
    features['PIR']     = features['IH'] / (abs(features['IL']) + 1e-8)
    features['Meu']     = float(np.mean(ppg_seg))
    features['ppg_std'] = float(np.std(ppg_seg))

    features['rise_time'] = (sys_peak - onset_idx) / fs

    half_amp = features['IH'] / 2.0
    above    = np.where(ppg_seg >= half_amp)[0]
    features['pulse_width'] = (above[-1] - above[0]) / fs if len(above) > 1 else 0.0

    search_d2_end = min(sys_peak + 1, len(ppg_d2))
    pos_apg, _    = find_peaks(ppg_d2[:search_d2_end])
    features['apg_a'] = float(ppg_d2[pos_apg[0]]) if len(pos_apg) > 0 \
                        else float(np.max(ppg_d2[:search_d2_end]))

    neg_apg, _ = find_peaks(-ppg_d2[:search_d2_end])
    features['apg_b'] = float(ppg_d2[neg_apg[0]]) if len(neg_apg) > 0 \
                        else float(np.min(ppg_d2[:search_d2_end]))

    features['aging_index']      = features['apg_b'] / (features['apg_a'] + 1e-8)

    total_area = float(np.trapezoid(ppg_seg))
    post_area  = float(np.trapezoid(ppg_seg[sys_peak:]))
    features['reflection_index'] = post_area / (total_area + 1e-8)
    features['auc']              = total_area / fs
    features['max_slope']        = float(np.max(ppg_d1))
    features['ppg_skew']         = float(skew(ppg_seg))
    features['ppg_kurt']         = float(kurtosis(ppg_seg))

    return features


# =============================================================================
#  FEATURE EXTRACTION — BLOOD GLUCOSE  (PPG-only, 21 features)
# =============================================================================

def extract_glucose_features(ppg_seg, fs=FS, hr=None):
    """
    Extract 21 PPG features for blood glucose estimation.

    Features 1–20:  time-domain, frequency-domain, 2nd-derivative waveform
                    features from the raw PPG / BVP signal.
    Feature 21:     heart rate (BPM) — from HR.csv during training,
                    from live PPG peaks during inference.

    Parameters
    ----------
    ppg_seg : np.ndarray  — one beat window of preprocessed PPG (125 samples)
    fs      : int         — sampling frequency in Hz
    hr      : float|None  — heart rate in BPM; estimated from PPG if None
    """
    if np.std(ppg_seg) < 1e-6:
        return np.zeros(21, dtype=np.float32)

    feats = []

    # ── Time-domain (8 features) ──────────────────────────────────────────────
    feats.append(float(np.mean(ppg_seg)))               # 1. mean
    feats.append(float(np.std(ppg_seg)))                # 2. std
    feats.append(float(skew(ppg_seg)))                  # 3. skewness
    feats.append(float(kurtosis(ppg_seg)))              # 4. kurtosis
    feats.append(float(np.sqrt(np.mean(ppg_seg**2))))   # 5. RMS
    feats.append(float(ppg_seg.max() - ppg_seg.min()))  # 6. range

    peaks,   _ = find_peaks(ppg_seg, height=np.mean(ppg_seg))
    troughs, _ = find_peaks(-ppg_seg)

    sys_amp = float(ppg_seg[peaks].mean())   if len(peaks)   > 0 else 0.0
    dia_amp = float(ppg_seg[troughs].mean()) if len(troughs) > 0 else 0.0
    feats.append(sys_amp)                               # 7. systolic amplitude
    feats.append(dia_amp)                               # 8. diastolic amplitude

    # ── Ratio features (2 features) ──────────────────────────────────────────
    feats.append(sys_amp / (abs(dia_amp) + 1e-8))       # 9.  amplitude ratio
    feats.append(float(np.trapezoid(ppg_seg) / fs))     # 10. area under curve

    # ── Frequency-domain (5 features) ────────────────────────────────────────
    N     = len(ppg_seg)
    freqs = fftfreq(N, 1.0 / fs)
    power = np.abs(fft(ppg_seg)) ** 2
    pos   = freqs > 0
    fp    = freqs[pos]
    pp    = power[pos]

    feats.append(float(fp[np.argmax(pp)]))                          # 11. dominant freq
    feats.append(float(pp[(fp >= 0.0) & (fp < 1.0)].sum()))        # 12. 0-1 Hz power
    feats.append(float(pp[(fp >= 1.0) & (fp < 3.0)].sum()))        # 13. 1-3 Hz power
    feats.append(float(pp[(fp >= 3.0) & (fp < 5.0)].sum()))        # 14. 3-5 Hz power
    feats.append(float(pp.sum()))                                   # 15. total power

    # ── 2nd derivative features (5 features) ─────────────────────────────────
    d1 = np.gradient(ppg_seg)
    d2 = np.gradient(d1)

    feats.append(float(np.max(d2)))                     # 16. 2nd deriv max
    feats.append(float(np.min(d2)))                     # 17. 2nd deriv min
    feats.append(float(np.max(d2) - np.min(d2)))        # 18. 2nd deriv range
    feats.append(float(np.std(d2)))                     # 19. 2nd deriv std
    feats.append(float(np.mean(np.abs(d2))))            # 20. mean abs 2nd deriv

    # ── HR feature (1 feature) ────────────────────────────────────────────────
    if hr is not None and np.isfinite(hr) and 30.0 <= hr <= 220.0:
        hr_feature = float(hr)
    else:
        ppg_n       = minmax_norm(ppg_seg)
        hr_peaks, _ = find_peaks(ppg_n, height=0.5, distance=int(fs * 0.4))
        if len(hr_peaks) >= 2:
            hr_feature = float(60.0 / np.mean(np.diff(hr_peaks) / fs))
        else:
            hr_feature = float(fp[np.argmax(pp)] * 60.0)
        hr_feature = float(np.clip(hr_feature, 30.0, 220.0))

    feats.append(hr_feature)                            # 21. heart rate (BPM)

    return np.array(feats, dtype=np.float32)
