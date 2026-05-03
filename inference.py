# =============================================================================
#  inference.py  —  PPG-only + PPG+ECG dual-mode version
#
#  Mode selection is automatic:
#    • ESP32 sends only "ppg"         → PPG-only model  (16 features)
#    • ESP32 sends "ppg" + "ecg"      → PPG+ECG model   (18 features)
#      If ECG quality check fails the PPG-only model is used as fallback.
#
#  Required model files in models/ (created by train_models.py):
#    PPG-only:  rf_bp_model_ppg.pkl, scaler_X_ppg.pkl, bp_feature_cols_ppg.pkl
#    PPG+ECG:   rf_bp_model_ecg.pkl, scaler_X_ecg.pkl, bp_feature_cols_ecg.pkl
#    Shared:    scaler_SBP.pkl, scaler_DBP.pkl
#    Glucose:   glucose_model.pkl, scaler_glu_X.pkl, scaler_glu_y.pkl
# =============================================================================

import numpy as np
import joblib
import os

from signal_processing import (
    preprocess_ppg,
    preprocess_ecg,
    segment_ppg_beats,
    segment_ecg_ppg_beats,
    check_ppg_quality,
    check_ecg_quality,
    extract_bp_features,
    extract_bp_features_with_ecg,
    extract_glucose_features,
)

FS = 125


# =============================================================================
#  PERSONALISED CALIBRATOR
# =============================================================================

class PersonalisedCalibrator:
    def __init__(self):
        self.sbp_slope  = 1.0;  self.sbp_offset  = 0.0
        self.dbp_slope  = 1.0;  self.dbp_offset  = 0.0
        self.glu_slope  = 1.0;  self.glu_offset  = 0.0
        self.calibrated = False
        self.n_samples  = 0

    def calibrate(self, pred_sbp, true_sbp, pred_dbp, true_dbp,
                  pred_glu=None, true_glu=None):
        pred_sbp = np.array(pred_sbp, dtype=float)
        true_sbp = np.array(true_sbp, dtype=float)
        pred_dbp = np.array(pred_dbp, dtype=float)
        true_dbp = np.array(true_dbp, dtype=float)

        if len(pred_sbp) < 2:
            self.sbp_slope  = 1.0
            self.sbp_offset = float(np.mean(true_sbp - pred_sbp))
            self.dbp_slope  = 1.0
            self.dbp_offset = float(np.mean(true_dbp - pred_dbp))
        else:
            self.sbp_slope, self.sbp_offset = np.polyfit(pred_sbp, true_sbp, 1)
            self.dbp_slope, self.dbp_offset = np.polyfit(pred_dbp, true_dbp, 1)

        if pred_glu is not None and true_glu is not None:
            pred_glu = np.array(pred_glu, dtype=float)
            true_glu = np.array(true_glu, dtype=float)
            if len(pred_glu) >= 2:
                self.glu_slope, self.glu_offset = np.polyfit(pred_glu, true_glu, 1)
            else:
                self.glu_slope  = 1.0
                self.glu_offset = float(np.mean(true_glu - pred_glu))

        self.calibrated = True
        self.n_samples  = len(pred_sbp)
        print(f"[Calibrator] Done — {self.n_samples} samples")

    def apply_bp(self, sbp, dbp):
        return (float(self.sbp_slope * sbp + self.sbp_offset),
                float(self.dbp_slope * dbp + self.dbp_offset))

    def apply_glucose(self, glucose):
        return float(self.glu_slope * glucose + self.glu_offset)

    def save(self, path="models/calibration.pkl"):
        joblib.dump({
            'sbp_slope': self.sbp_slope, 'sbp_offset': self.sbp_offset,
            'dbp_slope': self.dbp_slope, 'dbp_offset': self.dbp_offset,
            'glu_slope': self.glu_slope, 'glu_offset': self.glu_offset,
            'calibrated': self.calibrated, 'n_samples': self.n_samples,
        }, path)

    def load(self, path="models/calibration.pkl"):
        if not os.path.exists(path):
            return False
        d = joblib.load(path)
        self.sbp_slope  = d['sbp_slope'];  self.sbp_offset  = d['sbp_offset']
        self.dbp_slope  = d['dbp_slope'];  self.dbp_offset  = d['dbp_offset']
        self.glu_slope  = d['glu_slope'];  self.glu_offset  = d['glu_offset']
        self.calibrated = d['calibrated']; self.n_samples   = d['n_samples']
        return True


# =============================================================================
#  MAIN INFERENCE ENGINE
# =============================================================================

class MedWatchInference:

    MODELS_DIR = "models"

    def __init__(self):
        # PPG-only BP model (always required)
        self.bp_model_ppg        = None
        self.scaler_X_ppg        = None
        self.bp_feature_cols_ppg = None

        # PPG+ECG BP model (optional — loaded if present)
        self.bp_model_ecg        = None
        self.scaler_X_ecg        = None
        self.bp_feature_cols_ecg = None

        # Shared BP output scalers
        self.scaler_SBP = None
        self.scaler_DBP = None

        # Glucose model
        self.glu_model   = None
        self.scaler_glu_X = None
        self.scaler_glu_y = None

        self.calibrator = PersonalisedCalibrator()
        self._load_models()
        self.calibrator.load()

    def _load_models(self):
        # ── Required: PPG-only BP model ────────────────────────────────────
        required = {
            'bp_model_ppg':          'rf_bp_model_ppg.pkl',
            'scaler_X_ppg':          'scaler_X_ppg.pkl',
            'bp_feature_cols_ppg':   'bp_feature_cols_ppg.pkl',
            'scaler_SBP':            'scaler_SBP.pkl',
            'scaler_DBP':            'scaler_DBP.pkl',
            'glu_model':             'glucose_model.pkl',
            'scaler_glu_X':          'scaler_glu_X.pkl',
            'scaler_glu_y':          'scaler_glu_y.pkl',
        }
        for attr, filename in required.items():
            path = os.path.join(self.MODELS_DIR, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing: {path} — run train_models.py first."
                )
            setattr(self, attr, joblib.load(path))
            print(f"[Models] Loaded {filename}")

        # ── Optional: PPG+ECG BP model ────────────────────────────────────
        ecg_files = {
            'bp_model_ecg':          'rf_bp_model_ecg.pkl',
            'scaler_X_ecg':          'scaler_X_ecg.pkl',
            'bp_feature_cols_ecg':   'bp_feature_cols_ecg.pkl',
        }
        ecg_available = all(
            os.path.exists(os.path.join(self.MODELS_DIR, f))
            for f in ecg_files.values()
        )
        if ecg_available:
            for attr, filename in ecg_files.items():
                path = os.path.join(self.MODELS_DIR, filename)
                setattr(self, attr, joblib.load(path))
                print(f"[Models] Loaded {filename}")
            print("[Models] PPG+ECG model ready — will activate when ECG is received")
        else:
            print("[Models] PPG+ECG model not found — PPG-only mode only")
            print("[Models] Train with Harvard dataset to enable ECG mode")

    @property
    def ecg_model_available(self):
        return (self.bp_model_ecg is not None and
                self.scaler_X_ecg is not None and
                self.bp_feature_cols_ecg is not None)

    def predict(self, raw_ppg, raw_ecg=None, fs=FS):
        """
        Run inference on PPG (and optionally ECG) signals.

        Parameters
        ----------
        raw_ppg : list | np.ndarray  — raw PPG samples from ESP32 (≥125)
        raw_ecg : list | np.ndarray | None
                  Raw ECG samples from AD8232.  If provided and quality
                  checks pass, the PPG+ECG model is used automatically.
                  If None or quality check fails, falls back to PPG-only.
        fs      : int — sampling frequency in Hz (default 125)

        Returns
        -------
        dict: SBP, DBP, glucose, HR, mode, calibrated, quality
        """
        raw_ppg = np.array(raw_ppg, dtype=np.float32)

        # ── 1. PPG quality check ──────────────────────────────────────────
        ok, reason = check_ppg_quality(raw_ppg, fs=fs)
        if not ok:
            raise ValueError(f"Signal quality: {reason}")

        # ── 2. Preprocess PPG ─────────────────────────────────────────────
        ppg = preprocess_ppg(raw_ppg, fs=fs)

        # ── 3. Decide mode ────────────────────────────────────────────────
        use_ecg = False
        ecg     = None

        if raw_ecg is not None and self.ecg_model_available:
            raw_ecg = np.array(raw_ecg, dtype=np.float32)
            ecg_ok, ecg_reason = check_ecg_quality(raw_ecg, fs=fs)
            if ecg_ok:
                ecg     = preprocess_ecg(raw_ecg, fs=fs)
                use_ecg = True
            else:
                print(f"[Inference] ECG quality check failed ({ecg_reason}) "
                      f"— falling back to PPG-only")

        # ── 4. Segment and extract features ──────────────────────────────
        if use_ecg:
            sbp, dbp, hr_val = self._predict_bp_ecg(ppg, ecg, fs)
            mode = 'PPG+ECG'
        else:
            sbp, dbp, hr_val = self._predict_bp_ppg(ppg, fs)
            mode = 'PPG-only'

        # ── 5. Glucose (always PPG-only) ──────────────────────────────────
        glucose = self._predict_glucose(ppg, fs, hr_val)

        # ── 6. Personal calibration ───────────────────────────────────────
        if self.calibrator.calibrated:
            sbp, dbp = self.calibrator.apply_bp(sbp, dbp)
            glucose  = self.calibrator.apply_glucose(glucose)

        # ── 7. Physiological clamps ───────────────────────────────────────
        sbp     = float(np.clip(sbp,     70, 220))
        dbp     = float(np.clip(dbp,     40, 130))
        glucose = float(np.clip(glucose, 40, 500))
        if sbp <= dbp:
            sbp = dbp + 10.0

        return {
            'SBP':        round(sbp,     1),
            'DBP':        round(dbp,     1),
            'glucose':    round(glucose, 1),
            'HR':         round(hr_val,  0),
            'mode':       mode,
            'calibrated': self.calibrator.calibrated,
            'quality':    'good',
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _predict_bp_ppg(self, ppg, fs):
        """PPG-only BP prediction. Returns (sbp, dbp, hr)."""
        segments = segment_ppg_beats(ppg, fs=fs, window_sec=1.0)
        if len(segments) == 0:
            raise ValueError("Could not extract a clean beat window from PPG")

        ppg_seg = segments[len(segments) // 2]
        bp_feats = extract_bp_features(ppg_seg, fs=fs)
        if bp_feats is None:
            raise ValueError("BP feature extraction failed")

        feature_vector = np.array(
            [bp_feats.get(col, 0.0) for col in self.bp_feature_cols_ppg],
            dtype=np.float32
        )

        X_bp           = self.scaler_X_ppg.transform([feature_vector])
        bp_pred_scaled = self.bp_model_ppg.predict(X_bp)[0]

        sbp = float(self.scaler_SBP.inverse_transform([[bp_pred_scaled[0]]])[0][0])
        dbp = float(self.scaler_DBP.inverse_transform([[bp_pred_scaled[1]]])[0][0])
        hr  = float(bp_feats.get('HR', 0))

        return sbp, dbp, hr

    def _predict_bp_ecg(self, ppg, ecg, fs):
        """PPG+ECG BP prediction. Returns (sbp, dbp, hr)."""
        segments = segment_ecg_ppg_beats(ecg, ppg, fs=fs, window_sec=1.0)
        if len(segments) == 0:
            # ECG segmentation failed — fall back to PPG-only silently
            print("[Inference] ECG segmentation found no beats — using PPG-only")
            return self._predict_bp_ppg(ppg, fs)

        ecg_seg, ppg_seg = segments[len(segments) // 2]
        bp_feats = extract_bp_features_with_ecg(ppg_seg, ecg_seg, fs=fs)

        if bp_feats is None:
            print("[Inference] PTT extraction failed — using PPG-only")
            return self._predict_bp_ppg(ppg, fs)

        feature_vector = np.array(
            [bp_feats.get(col, 0.0) for col in self.bp_feature_cols_ecg],
            dtype=np.float32
        )

        X_bp           = self.scaler_X_ecg.transform([feature_vector])
        bp_pred_scaled = self.bp_model_ecg.predict(X_bp)[0]

        sbp = float(self.scaler_SBP.inverse_transform([[bp_pred_scaled[0]]])[0][0])
        dbp = float(self.scaler_DBP.inverse_transform([[bp_pred_scaled[1]]])[0][0])
        hr  = float(bp_feats.get('HR', 0))

        return sbp, dbp, hr

    def _predict_glucose(self, ppg, fs, hr):
        """Glucose prediction (always PPG-only). Returns glucose in mg/dL."""
        segments = segment_ppg_beats(ppg, fs=fs, window_sec=1.0)
        if len(segments) == 0:
            raise ValueError("Could not extract a clean beat window for glucose")

        ppg_seg = segments[len(segments) // 2]
        glu_feats = extract_glucose_features(ppg_seg, fs=fs, hr=hr)

        X_glu      = self.scaler_glu_X.transform([glu_feats])
        glu_scaled = self.glu_model.predict(X_glu)[0]
        glucose    = float(self.scaler_glu_y.inverse_transform([[glu_scaled]])[0][0])

        return glucose

    # ── Public calibration API ────────────────────────────────────────────────

    def calibrate(self, pred_sbp, true_sbp, pred_dbp, true_dbp,
                  pred_glu=None, true_glu=None):
        self.calibrator.calibrate(
            pred_sbp, true_sbp, pred_dbp, true_dbp, pred_glu, true_glu
        )
        self.calibrator.save()
