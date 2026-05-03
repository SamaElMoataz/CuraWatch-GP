# =============================================================================
#  train_models.py  —  PPG-only + PPG+ECG + Glucose
#
#  BP PPG-only model  — PPG-BP Figshare dataset  (Liang et al. 2018)
#    219 subjects, 657 records, wrist PPG at 1000 Hz, cuff BP labels
#
#  BP PPG+ECG model  — MIMIC-III Waveform Database (Harvard Dataverse)
#    ECG + PPG paired recordings, ABP-derived SBP/DBP labels
#    All waveforms are already at 125 Hz — no resampling needed.
#
#  Glucose model  — BIG IDEAs Lab Glycemic Variability (PhysioNet)
#    16 subjects, wrist PPG (BVP) + CGM glucose
# =============================================================================

import numpy as np
import pandas as pd
import joblib
import os
import glob
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

os.makedirs('models', exist_ok=True)

# =============================================================================
#  FEATURE COLUMN DEFINITIONS
# =============================================================================

FEATURE_COLS_PPG = [
    'HR', 'IH', 'IL', 'PIR', 'Meu', 'ppg_std',
    'rise_time', 'pulse_width',
    'apg_a', 'apg_b', 'aging_index',
    'reflection_index', 'auc', 'max_slope',
    'ppg_skew', 'ppg_kurt',
]

FEATURE_COLS_ECG = [
    'PTT', 'WN',                            # ECG-derived features
    'HR', 'IH', 'IL', 'PIR', 'Meu', 'ppg_std',
    'rise_time', 'pulse_width',
    'apg_a', 'apg_b', 'aging_index',
    'reflection_index', 'auc', 'max_slope',
    'ppg_skew', 'ppg_kurt',
]


# =============================================================================
#  HELPER: Load PPG-BP Figshare label file
# =============================================================================

def _load_ppg_bp_labels(data_dir):
    """
    Load SBP/DBP labels from PPG-BP.xlsx.
    Handles multi-row header by scanning rows 0-5.
    Returns DataFrame with columns: subject_id, SBP, DBP  or None.
    """
    label_path = os.path.join(data_dir, 'PPG-BP.xlsx')
    if not os.path.exists(label_path):
        print(f"[PPGBP] Label file not found: {label_path}")
        return None

    print(f"[PPGBP] Loading label file: PPG-BP.xlsx")

    for header_row in range(6):
        try:
            df         = pd.read_excel(label_path, header=header_row)
            cols_upper = [str(c).upper() for c in df.columns]

            has_sbp = any('SBP' in c or 'SYSTOLIC'  in c for c in cols_upper)
            has_dbp = any('DBP' in c or 'DIASTOLIC' in c for c in cols_upper)

            if not (has_sbp and has_dbp):
                continue

            print(f"[PPGBP] Found BP columns at header row {header_row}")

            sbp_col = next(c for c in df.columns
                           if 'SBP' in str(c).upper() or 'SYSTOLIC'  in str(c).upper())
            dbp_col = next(c for c in df.columns
                           if 'DBP' in str(c).upper() or 'DIASTOLIC' in str(c).upper())
            id_col  = next((c for c in df.columns
                            if 'SUBJECT' in str(c).upper()), df.columns[0])

            df = df[[id_col, sbp_col, dbp_col]].copy()
            df.columns = ['subject_id', 'SBP', 'DBP']
            df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce')
            df['SBP']        = pd.to_numeric(df['SBP'],        errors='coerce')
            df['DBP']        = pd.to_numeric(df['DBP'],        errors='coerce')
            df = df.dropna().reset_index(drop=True)
            df['subject_id'] = df['subject_id'].astype(int)

            return df

        except Exception:
            continue

    print("[PPGBP] Could not parse PPG-BP.xlsx.")
    return None


# =============================================================================
#  SHARED BP TRAINING HELPER
# =============================================================================

def _train_and_save_bp(X, y, feature_cols, mode, dataset_name,
                        subject_ids=None):
    """
    Shared BP training, evaluation, and saving logic.

    Parameters
    ----------
    X            : np.ndarray  (n_samples, n_features)
    y            : np.ndarray  (n_samples, 2)  [SBP, DBP]
    feature_cols : list[str]   in same order as X columns
    mode         : 'ppg_only' | 'ppg_ecg'
    dataset_name : str  used in saved filenames
    subject_ids  : np.ndarray  for subject-level split; random split if None
    """
    valid = ~np.any(np.isnan(X), axis=1)
    X     = X[valid]
    y     = y[valid]
    if subject_ids is not None:
        subject_ids = subject_ids[valid]

    tag = dataset_name.upper()
    print(f"[{tag}] Samples after NaN removal: {len(X)}")

    if len(X) < 50:
        print(f"[{tag}] Not enough samples. Aborting.")
        return False

    # ── Train / test split ─────────────────────────────────────────────────
    if subject_ids is not None:
        unique_subjects = np.unique(subject_ids)
        np.random.seed(42)
        np.random.shuffle(unique_subjects)
        n_test         = max(1, int(len(unique_subjects) * 0.2))
        test_subjects  = unique_subjects[:n_test]
        train_subjects = unique_subjects[n_test:]

        train_mask = np.isin(subject_ids, train_subjects)
        test_mask  = np.isin(subject_ids, test_subjects)

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        print(f"[{tag}] Subject-level split:")
        print(f"  Train: {len(X_tr)} samples ({len(train_subjects)} subjects)")
        print(f"  Test:  {len(X_te)} samples ({len(test_subjects)} subjects)")
    else:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        print(f"[{tag}] Random split: Train={len(X_tr)}  Test={len(X_te)}")

    # ── Normalize ──────────────────────────────────────────────────────────
    scaler_X   = StandardScaler()
    scaler_SBP = StandardScaler()
    scaler_DBP = StandardScaler()

    X_tr_sc   = scaler_X.fit_transform(X_tr)
    X_te_sc   = scaler_X.transform(X_te)
    sbp_tr_sc = scaler_SBP.fit_transform(y_tr[:, 0:1])
    dbp_tr_sc = scaler_DBP.fit_transform(y_tr[:, 1:2])
    y_tr_sc   = np.hstack([sbp_tr_sc, dbp_tr_sc])

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"[{tag}] Training Random Forest (500 trees, mode={mode})...")
    rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=500, random_state=2, n_jobs=-1)
    )
    rf.fit(X_tr_sc, y_tr_sc)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred   = rf.predict(X_te_sc)
    sbp_pred = scaler_SBP.inverse_transform(y_pred[:, 0:1]).flatten()
    dbp_pred = scaler_DBP.inverse_transform(y_pred[:, 1:2]).flatten()
    sbp_true = y_te[:, 0]
    dbp_true = y_te[:, 1]

    mae_sbp = mean_absolute_error(sbp_true, sbp_pred)
    mae_dbp = mean_absolute_error(dbp_true, dbp_pred)
    std_sbp = np.std(sbp_true - sbp_pred)
    std_dbp = np.std(dbp_true - dbp_pred)

    print(f"\n[{tag}] Results ({mode}):")
    print(f"  SBP  MAE={mae_sbp:.2f} mmHg  STD={std_sbp:.2f} mmHg")
    print(f"  DBP  MAE={mae_dbp:.2f} mmHg  STD={std_dbp:.2f} mmHg")
    print("  AAMI Pass if MAE<=5 and STD<=8")

    # ── Save ───────────────────────────────────────────────────────────────
    suffix = f"{dataset_name}_{mode}"
    joblib.dump(rf,           f'models/rf_bp_model_{suffix}.pkl')
    joblib.dump(scaler_X,     f'models/scaler_X_{suffix}.pkl')
    joblib.dump(feature_cols, f'models/bp_feature_cols_{suffix}.pkl')

    # Standard names loaded by inference.py
    if mode == 'ppg_only':
        joblib.dump(rf,           'models/rf_bp_model_ppg.pkl')
        joblib.dump(scaler_X,     'models/scaler_X_ppg.pkl')
        joblib.dump(feature_cols, 'models/bp_feature_cols_ppg.pkl')
        joblib.dump(scaler_SBP,   'models/scaler_SBP.pkl')
        joblib.dump(scaler_DBP,   'models/scaler_DBP.pkl')
    elif mode == 'ppg_ecg':
        joblib.dump(rf,           'models/rf_bp_model_ecg.pkl')
        joblib.dump(scaler_X,     'models/scaler_X_ecg.pkl')
        joblib.dump(feature_cols, 'models/bp_feature_cols_ecg.pkl')
        # ECG model uses the same scaler_SBP & scaler_DBP as the PPG model
        # so that inference.py can share them.  Only overwrite if ppg_only
        # was already saved (they will be identical targets anyway).
        if not os.path.exists('models/scaler_SBP.pkl'):
            joblib.dump(scaler_SBP, 'models/scaler_SBP.pkl')
            joblib.dump(scaler_DBP, 'models/scaler_DBP.pkl')

    print(f"[{tag}] Saved to models/")
    return True


# =============================================================================
#  BP MODEL — PPG-only  (Figshare PPG-BP dataset)
# =============================================================================

def train_bp_model_ppg():
    """
    Train PPG-only BP model using PPG-BP Figshare dataset.
    Extracts 16 morphology features per record.
    """
    from scipy.signal import resample
    from signal_processing import (
        preprocess_ppg, segment_ppg_beats, extract_bp_features
    )
    from tqdm import tqdm

    print("\n" + "=" * 60)
    print("TRAINING BP MODEL — PPG-only (Figshare)")
    print("=" * 60)

    data_dir = 'data/ppg_bp'
    subj_dir = os.path.join(data_dir, '0_subject')

    if not os.path.exists(data_dir):
        print(f"[PPGBP] Folder not found: {data_dir}")
        return False

    if not os.path.exists(subj_dir):
        print(f"[PPGBP] Subject folder not found: {subj_dir}")
        return False

    labels = _load_ppg_bp_labels(data_dir)
    if labels is None:
        return False

    all_txt_files = glob.glob(os.path.join(subj_dir, '*.txt'))
    print(f"[PPGBP] Found {len(all_txt_files)} PPG text files")

    if len(all_txt_files) == 0:
        print("[PPGBP] No .txt files found")
        return False

    subject_files = {}
    for fpath in all_txt_files:
        name  = os.path.splitext(os.path.basename(fpath))[0]
        parts = name.rsplit('_', 1)
        if len(parts) != 2:
            continue
        try:
            subj_id = int(parts[0])
            seg_num = int(parts[1])
        except ValueError:
            continue
        subject_files.setdefault(subj_id, []).append((seg_num, fpath))

    for sid in subject_files:
        subject_files[sid].sort()

    FS_ORIGINAL = 1000
    FS_TARGET   = 125

    X_list      = []
    y_list      = []
    subject_ids = []
    n_skipped   = 0

    print("[PPGBP] Extracting features...")

    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        subj_id = int(row['subject_id'])
        sbp     = float(row['SBP'])
        dbp     = float(row['DBP'])

        if not (75 <= sbp <= 200 and 40 <= dbp <= 130 and sbp > dbp):
            n_skipped += 1
            continue

        if subj_id not in subject_files:
            n_skipped += 1
            continue

        for seg_num, seg_path in subject_files[subj_id]:
            try:
                ppg_raw = np.loadtxt(seg_path).flatten().astype(np.float32)

                if len(ppg_raw) < 500:
                    continue

                n_target = int(len(ppg_raw) * FS_TARGET / FS_ORIGINAL)
                ppg_125  = resample(ppg_raw, n_target).astype(np.float32)

                if len(ppg_125) < FS_TARGET:
                    continue

                ppg   = preprocess_ppg(ppg_125, fs=FS_TARGET)
                beats = segment_ppg_beats(ppg, fs=FS_TARGET, window_sec=1.0)

                if len(beats) == 0:
                    continue

                ppg_seg = beats[len(beats) // 2]
                feats   = extract_bp_features(ppg_seg, fs=FS_TARGET)

                if feats is None:
                    continue

                X_list.append([feats.get(col, 0.0) for col in FEATURE_COLS_PPG])
                y_list.append([sbp, dbp])
                subject_ids.append(subj_id)

            except Exception:
                continue

    print(f"[PPGBP] Skipped: {n_skipped}  Valid: {len(X_list)}")

    if len(X_list) == 0:
        print("[PPGBP] No valid segments extracted.")
        return False

    X           = np.array(X_list,      dtype=np.float32)
    y           = np.array(y_list,      dtype=np.float32)
    subject_ids = np.array(subject_ids, dtype=np.int32)

    return _train_and_save_bp(
        X, y, FEATURE_COLS_PPG,
        mode='ppg_only',
        dataset_name='ppg_bp',
        subject_ids=subject_ids
    )


# =============================================================================
#  BP MODEL — PPG+ECG  (Harvard Dataverse MIMIC-III subset)
# =============================================================================

def train_bp_model_ecg():
    """
    Each file contains 30 segments of 30 seconds at 125 Hz (3750 samples).
    Labels are SBP/DBP in mmHg derived from the ABP waveform.

    If the dataset is not present this function exits gracefully and returns
    False & the system continues with the PPG-only model.
    """
    from signal_processing import (
        preprocess_ppg, preprocess_ecg,
        segment_ecg_ppg_beats,
        extract_bp_features_with_ecg
    )
    from tqdm import tqdm
    import ast

    print("\n" + "=" * 60)
    print("TRAINING BP MODEL — PPG+ECG")
    print("=" * 60)

    data_dir = 'data/harvard_bp'
    FS       = 125   # all waveforms are already at 125 Hz

    if not os.path.exists(data_dir):
        print(f" [PPGECGBP] Folder not found: {data_dir}")
        print(" [PPGECGBP] Skipping ECG model — PPG-only model will be used")
        return False

    # ── Load the predefined subject splits ────────────────────────────────
    def _load_subject_list(txt_path):
        """Read a file containing a Python list literal of subject ID strings."""
        if not os.path.exists(txt_path):
            return None
        with open(txt_path, 'r') as fh:
            content = fh.read().strip()
        try:
            ids = ast.literal_eval(content)   # e.g. ['p000188', 'p000333', ...]
            return [s.strip() for s in ids]
        except Exception:
            # Fallback: one ID per line
            return [ln.strip() for ln in content.splitlines() if ln.strip()]

    train_ids = _load_subject_list(os.path.join(data_dir, 'train_subjects.txt'))
    val_ids   = _load_subject_list(os.path.join(data_dir, 'val_subjects.txt'))
    test_ids  = _load_subject_list(os.path.join(data_dir, 'test_subjects.txt'))

    if train_ids is None:
        print(" [PPGECGBP] train_subjects.txt not found")
        # Discover all subjects from available label files
        label_files = glob.glob(os.path.join(data_dir, '*_labels.npy'))
        all_ids     = sorted(
            os.path.basename(f).replace('_labels.npy', '') for f in label_files
        )
        np.random.seed(42)
        np.random.shuffle(all_ids)
        n_test     = max(1, int(len(all_ids) * 0.2))
        n_val      = max(1, int(len(all_ids) * 0.1))
        test_ids   = all_ids[:n_test]
        val_ids    = all_ids[n_test:n_test + n_val]
        train_ids  = all_ids[n_test + n_val:]
    else:
        if val_ids  is None: val_ids  = []
        if test_ids is None: test_ids = []

    print(f" [PPGECGBP] Subjects — train: {len(train_ids)}  "
          f"val: {len(val_ids)}  test: {len(test_ids)}")

    # Training uses train + val subjects; test subjects are held out entirely.
    fit_ids  = list(train_ids) + list(val_ids)
    held_ids = list(test_ids)

    # ── Feature extraction ────────────────────────────────────────────────
    X_list   = []
    y_list   = []
    sid_list = []    # numeric subject index for _train_and_save_bp split logic

    # We process fit + held subjects but tag them so we can do the split later.
    # Pass all subject IDs and let _train_and_save_bp handle subject-level split.
    all_ids_ordered = fit_ids + held_ids
    fit_set         = set(fit_ids)

    print(f" [PPGECGBP] Extracting features from "
          f"{len(all_ids_ordered)} subjects")

    for subj_str in tqdm(all_ids_ordered, desc="Processing subjects"):
        ppg_path    = os.path.join(data_dir, f'{subj_str}_ppg.npy')
        ecg_path    = os.path.join(data_dir, f'{subj_str}_ecg.npy')
        labels_path = os.path.join(data_dir, f'{subj_str}_labels.npy')

        if not all(os.path.exists(p) for p in [ppg_path, ecg_path, labels_path]):
            continue

        try:
            ppg_all    = np.load(ppg_path)    # (30, 3750)
            ecg_all    = np.load(ecg_path)    # (30, 3750)
            labels_all = np.load(labels_path) # (30, 2)  [SBP, DBP]

            # Validate expected shapes
            if ppg_all.ndim != 2 or ppg_all.shape[1] != 3750:
                continue
            if ecg_all.shape != ppg_all.shape:
                continue
            if labels_all.ndim != 2 or labels_all.shape[1] != 2:
                continue

            # Numeric subject ID (strip leading 'p' and parse integer)
            subj_id = int(''.join(filter(str.isdigit, subj_str)) or 0)

            # Process each of the 30 segments independently
            for seg_idx in range(ppg_all.shape[0]):
                sbp = float(labels_all[seg_idx, 0])
                dbp = float(labels_all[seg_idx, 1])

                # Physiological range check
                if not (75 <= sbp <= 200 and 40 <= dbp <= 130 and sbp > dbp):
                    continue

                ppg_raw = ppg_all[seg_idx].astype(np.float32)  # 3750 samples
                ecg_raw = ecg_all[seg_idx].astype(np.float32)  # 3750 samples

                # Skip flat / near-flat segments (likely artifact or dropout)
                if np.std(ppg_raw) < 1e-6 or np.std(ecg_raw) < 1e-6:
                    continue

                # ── Preprocess ────────────────────────────────────────
                ppg = preprocess_ppg(ppg_raw, fs=FS)
                ecg = preprocess_ecg(ecg_raw, fs=FS)

                # ── Segment — aligned ECG+PPG beat windows ────────────
                segments = segment_ecg_ppg_beats(
                    ecg, ppg, fs=FS, window_sec=1.0
                )

                if len(segments) == 0:
                    continue

                # Use the middle beat for a representative feature vector
                ecg_seg, ppg_seg = segments[len(segments) // 2]

                # ── Extract 18 features (16 PPG + PTT + WN) ──────────
                feats = extract_bp_features_with_ecg(
                    ppg_seg, ecg_seg, fs=FS
                )

                if feats is None:
                    continue

                X_list.append([feats.get(col, 0.0) for col in FEATURE_COLS_ECG])
                y_list.append([sbp, dbp])
                sid_list.append(subj_id)

        except Exception as e:
            print(f" [PPGECGBP] Error on {subj_str}: {e}")
            continue

    if len(X_list) == 0:
        print(" [PPGECGBP] No valid windows extracted.")
        return False

    X           = np.array(X_list,  dtype=np.float32)
    y           = np.array(y_list,  dtype=np.float32)
    subject_ids = np.array(sid_list, dtype=np.int32)

    print(f"\n [PPGECGBP] Total segments : {len(X)}")
    print(f" [PPGECGBP] Unique subjects : {len(np.unique(subject_ids))}")

    return _train_and_save_bp(
        X, y, FEATURE_COLS_ECG,
        mode='ppg_ecg',
        dataset_name='harvard_bp',
        subject_ids=subject_ids
    )


# =============================================================================
#  GLUCOSE MODEL — BIG IDEAs Lab (PhysioNet)
# =============================================================================

def train_glucose_model():
    """
    Train blood glucose model using BIG IDEAs Lab PhysioNet dataset.
    """
    from signal_processing import extract_glucose_features
    from scipy.signal import resample
    from tqdm import tqdm

    print("\n" + "=" * 60)
    print("TRAINING BLOOD GLUCOSE MODEL")
    print("Dataset: BIG IDEAs Lab Glycemic Variability (PhysioNet)")
    print("=" * 60)

    data_dir = 'data/bigideas'

    if not os.path.exists(data_dir):
        print(f"[BIGIDEAS] Dataset folder not found: {data_dir}")
        return False

    FS_BVP       = 64
    FS_TARGET    = 125
    WINDOW       = 125
    STEP         = 62
    MAX_TIME_GAP = 300

    X_list      = []
    y_list      = []
    subject_ids = []

    subject_folders = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
    ])

    if len(subject_folders) == 0:
        print(f"[BIGIDEAS] No subject folders found in {data_dir}")
        return False

    print(f"[BIGIDEAS] Found {len(subject_folders)} subject folders")

    for subj_folder in tqdm(subject_folders, desc="Processing subjects"):
        subj_id   = int(subj_folder)
        subj_path = os.path.join(data_dir, subj_folder)

        bvp_path    = os.path.join(subj_path, 'BVP.csv')
        dexcom_path = os.path.join(subj_path, 'Dexcom.csv')

        if not os.path.exists(bvp_path) or not os.path.exists(dexcom_path):
            continue

        try:
            bvp_raw = pd.read_csv(bvp_path, header=None)

            # BVP.csv format: two columns — [MM:SS.f timestamp, PPG value]
            # Column 0 is a relative "MM:SS.f" string (e.g. "28:50.0"),
            # Column 1 is the actual BVP/PPG sample value.
            if bvp_raw.shape[1] >= 2:
                # Parse MM:SS.f timestamps from column 0 → seconds
                def _mmss_to_sec(s):
                    try:
                        parts = str(s).split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    except Exception:
                        return np.nan

                ts_sec = bvp_raw.iloc[:, 0].apply(_mmss_to_sec).values
                bvp_values = bvp_raw.iloc[:, 1].values.astype(np.float32)

                # Drop rows where timestamp or value could not be parsed
                valid_mask = ~(np.isnan(ts_sec) | np.isnan(bvp_values))
                ts_sec     = ts_sec[valid_mask]
                bvp_values = bvp_values[valid_mask]

                bvp_start_ts = ts_sec[0] if len(ts_sec) > 0 else 0.0
            else:
                # Fallback: single-column file, treat values as raw PPG
                bvp_values   = bvp_raw.iloc[:, 0].values.astype(np.float32)
                bvp_start_ts = 0.0

            if len(bvp_values) < FS_BVP * 60:
                continue

            bvp_times = bvp_start_ts + np.arange(len(bvp_values)) / FS_BVP

            dex = pd.read_csv(dexcom_path)

            glu_col = next(
                (c for c in dex.columns
                 if 'glucose' in str(c).lower() or 'mg' in str(c).lower()),
                None
            )
            ts_col = next(
                (c for c in dex.columns
                 if 'time' in str(c).lower()
                 or 'timestamp' in str(c).lower()),
                None
            )

            if glu_col is None or ts_col is None:
                continue

            event_col = next(
                (c for c in dex.columns
                 if 'event' in str(c).lower() or 'type' in str(c).lower()),
                None
            )
            if event_col is not None and 'EGV' in dex[event_col].values:
                dex = dex[dex[event_col] == 'EGV'].copy()

            dex[glu_col] = pd.to_numeric(dex[glu_col], errors='coerce')
            dex = dex.dropna(subset=[glu_col]).reset_index(drop=True)

            try:
                dex['ts_unix'] = (
                    pd.to_datetime(dex[ts_col]).astype(np.int64) // 10**9
                )
            except Exception:
                continue

            dex = dex.sort_values('ts_unix').reset_index(drop=True)
            if len(dex) < 5:
                continue

            # Normalize Dexcom timestamps to seconds relative to their own
            # start, so they share the same relative time axis as bvp_times
            # (which is also relative, in seconds from the recording start).
            glucose_times_abs = dex['ts_unix'].values.astype(np.float64)
            glucose_times     = glucose_times_abs - glucose_times_abs[0]

            glucose_values = dex[glu_col].values.astype(np.float32)

            # Removing Outliers
            valid_glu      = (glucose_values >= 40) & (glucose_values <= 500)
            glucose_times  = glucose_times[valid_glu]
            glucose_values = glucose_values[valid_glu]

            if len(glucose_values) < 3:
                continue

            # Also normalize bvp_times to start at 0 so both axes match
            bvp_times = bvp_times - bvp_times[0]

            n_target      = int(len(bvp_values) * FS_TARGET / FS_BVP)
            bvp_125       = resample(bvp_values, n_target).astype(np.float32)
            bvp_times_125 = np.linspace(
                bvp_times[0], bvp_times[-1], len(bvp_125)
            )

            for start in range(0, len(bvp_125) - WINDOW, STEP):
                end     = start + WINDOW
                ppg_win = bvp_125[start:end]

                if np.std(ppg_win) < 1e-6:
                    continue

                win_time     = bvp_times_125[start + WINDOW // 2]
                time_diffs   = np.abs(glucose_times - win_time)
                nearest_idx  = np.argmin(time_diffs)

                if time_diffs[nearest_idx] > MAX_TIME_GAP:
                    continue

                glucose_label = float(glucose_values[nearest_idx])
                feats         = extract_glucose_features(ppg_win, fs=FS_TARGET)

                if np.any(np.isnan(feats)):
                    continue

                X_list.append(feats)
                y_list.append(glucose_label)
                subject_ids.append(subj_id)

        except Exception as e:
            print(f"[BIGIDEAS] Error on subject {subj_folder}: {e}")
            continue

    if len(X_list) == 0:
        print("[BIGIDEAS] No valid windows extracted.")
        return False

    X           = np.array(X_list,      dtype=np.float32)
    y           = np.array(y_list,      dtype=np.float32)
    subject_ids = np.array(subject_ids, dtype=np.int32)

    print(f"\n[BIGIDEAS] Total windows: {len(X)}")
    print(f"[BIGIDEAS] Subjects: {len(np.unique(subject_ids))}")
    print(f"[BIGIDEAS] Glucose: {y.min():.1f} – {y.max():.1f} mg/dL")

    # Subject-level split
    unique_subjects = np.unique(subject_ids)
    np.random.seed(42)
    np.random.shuffle(unique_subjects)
    n_test         = max(1, int(len(unique_subjects) * 0.2))
    test_subjects  = unique_subjects[:n_test]
    train_subjects = unique_subjects[n_test:]

    train_mask = np.isin(subject_ids, train_subjects)
    test_mask  = np.isin(subject_ids, test_subjects)

    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    print(f"[BIGIDEAS] Train: {len(X_tr)} windows ({len(train_subjects)} subjects)")
    print(f"[BIGIDEAS] Test:  {len(X_te)} windows ({len(test_subjects)} subjects)")

    scaler_glu_X = StandardScaler()
    scaler_glu_y = StandardScaler()

    X_tr_sc = scaler_glu_X.fit_transform(X_tr)
    X_te_sc = scaler_glu_X.transform(X_te)
    y_tr_sc = scaler_glu_y.fit_transform(y_tr.reshape(-1, 1)).flatten()

    print("[BIGIDEAS] Training Gradient Boosting Regressor...")
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_tr_sc, y_tr_sc)

    y_pred = scaler_glu_y.inverse_transform(
        gb.predict(X_te_sc).reshape(-1, 1)
    ).flatten()

    mae_glu = mean_absolute_error(y_te, y_pred)
    std_glu = np.std(y_te - y_pred)
    within_20pct = np.mean(
        np.abs(y_pred - y_te) / (np.abs(y_te) + 1e-8) < 0.20
    ) * 100

    print(f"\n[BIGIDEAS] MAE={mae_glu:.1f} mg/dL  STD={std_glu:.1f} mg/dL")
    print(f"[BIGIDEAS] Within 20% (Clarke Zone A): {within_20pct:.1f}%")

    joblib.dump(gb,           'models/glucose_model.pkl')
    joblib.dump(scaler_glu_X, 'models/scaler_glu_X.pkl')
    joblib.dump(scaler_glu_y, 'models/scaler_glu_y.pkl')
    print("[BIGIDEAS] Saved to models/")
    return True


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MEDWATCH MODEL TRAINING")
    print("=" * 60)

    bp_ppg_ok = train_bp_model_ppg()
    bp_ecg_ok = train_bp_model_ecg()
    glu_ok    = train_glucose_model()

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print(f"  BP PPG-only model: {'OK' if bp_ppg_ok else 'SKIPPED'}")
    print(f"  BP PPG+ECG model:  {'OK' if bp_ecg_ok else 'SKIPPED (dataset missing or no valid data)'}")
    print(f"  Glucose model:     {'OK' if glu_ok    else 'SKIPPED'}")
    print("=" * 60)

    if bp_ppg_ok:
        print("\nPPG-only model ready")
    if bp_ecg_ok:
        print("PPG+ECG model ready")
    if bp_ppg_ok and glu_ok:
        print("All models trained")

    if not bp_ppg_ok:
        print("\nBP PPG-only model not trained")

    if not bp_ecg_ok:
        print("\nBP PPG+ECG model not trained (optional)")

    if not glu_ok:
        print("\nGlucose model not trained")
