# =============================================================================
#  train_models.py  —  PPG-only version using PPG-BP Figshare dataset
#
#  BP Dataset: PPG-BP Figshare (Liang et al., Scientific Data 2018)
#    - 219 subjects, 657 PPG records, cuff BP labels
#    - PPG only (no ECG), raw waveforms at 1000 Hz resampled to 125 Hz
#    - Download: https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299
#    - Extract to: data/ppg_bp/
#      Expected structure:
#        data/ppg_bp/
#        ├── PPG-BP.xlsx          (label file — multi-row header)
#        └── 0_subject/
#            ├── 2_1.txt          (subject 2, segment 1)
#            ├── 2_2.txt
#            ├── 2_3.txt
#            ├── 3_1.txt
#            └── ... up to 419_3.txt
#
#  Glucose Dataset: BIG IDEAs Lab Glycemic Variability (PhysioNet)
#    - Download: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/
#    - Requires free PhysioNet account
#    - Extract to: data/bigideas/
#
#  Run once before starting server.py:
#      python train_models.py
# =============================================================================

import numpy as np
import pandas as pd
import joblib
import os
import glob
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

os.makedirs('models', exist_ok=True)


# =============================================================================
#  HELPER: Load PPG-BP label file
#  PPG-BP.xlsx has a multi-row header. This function scans rows 0-5 looking
#  for the row that contains Systolic/Diastolic column names.
#  Only PPG-BP.xlsx is used — Table 1.xlsx is ignored.
# =============================================================================

def _load_ppg_bp_labels(data_dir):
    """
    Load SBP/DBP labels from PPG-BP.xlsx only.
    Handles multi-row header by scanning every row for Systolic/Diastolic.

    Returns DataFrame with columns: subject_id, SBP, DBP
    Returns None if file not found or cannot be parsed.
    """
    label_path = os.path.join(data_dir, 'PPG-BP.xlsx')

    if not os.path.exists(label_path):
        print(f"[PPGBP] Label file not found: {label_path}")
        print("[PPGBP] Expected: data/ppg_bp/PPG-BP.xlsx")
        return None

    print(f"[PPGBP] Loading label file: PPG-BP.xlsx")

    # Scan rows 0 through 5 to find the actual header row
    for header_row in range(6):
        try:
            df = pd.read_excel(label_path, header=header_row)

            cols_upper = [str(c).upper() for c in df.columns]

            # Match both short ('SBP') and full word ('SYSTOLIC') column names
            has_sbp = any(
                'SBP' in c or 'SYSTOLIC' in c for c in cols_upper
            )
            has_dbp = any(
                'DBP' in c or 'DIASTOLIC' in c for c in cols_upper
            )

            if has_sbp and has_dbp:
                print(f"[PPGBP] Found BP columns at header row {header_row}")
                print(f"[PPGBP] Columns: {df.columns.tolist()}")

                # Find exact column names
                sbp_col = next(
                    c for c in df.columns
                    if 'SBP' in str(c).upper() or 'SYSTOLIC' in str(c).upper()
                )
                dbp_col = next(
                    c for c in df.columns
                    if 'DBP' in str(c).upper() or 'DIASTOLIC' in str(c).upper()
                )

                # Subject ID column — look for 'subject' keyword first
                id_col = next(
                    (c for c in df.columns
                     if 'SUBJECT' in str(c).upper()),
                    df.columns[0]
                )

                # Extract and clean the three relevant columns
                df = df[[id_col, sbp_col, dbp_col]].copy()
                df.columns = ['subject_id', 'SBP', 'DBP']

                df['subject_id'] = pd.to_numeric(
                    df['subject_id'], errors='coerce'
                )
                df['SBP'] = pd.to_numeric(df['SBP'], errors='coerce')
                df['DBP'] = pd.to_numeric(df['DBP'], errors='coerce')
                df = df.dropna().reset_index(drop=True)
                df['subject_id'] = df['subject_id'].astype(int)

                print(f"[PPGBP] Valid label rows: {len(df)}")
                print(f"[PPGBP] Subject ID range: "
                      f"{df['subject_id'].min()} – {df['subject_id'].max()}")
                print(f"[PPGBP] SBP range: "
                      f"{df['SBP'].min():.1f} – {df['SBP'].max():.1f} mmHg")
                print(f"[PPGBP] DBP range: "
                      f"{df['DBP'].min():.1f} – {df['DBP'].max():.1f} mmHg")
                return df

        except Exception as e:
            continue

    print("[PPGBP] Could not parse PPG-BP.xlsx.")
    print("[PPGBP] Tried header rows 0-5, none contained SBP/DBP columns.")
    return None


# =============================================================================
#  SHARED TRAINING HELPER
# =============================================================================

def _train_and_save_bp(X, y, FEATURE_COLS, mode, dataset_name,
                        subject_ids=None):
    """
    Shared BP training, evaluation, and saving logic.

    Parameters
    ----------
    X            : np.ndarray  (n_samples, n_features)
    y            : np.ndarray  (n_samples, 2)  columns = [SBP, DBP]
    FEATURE_COLS : list of str  in same order as X columns
    mode         : 'ppg_only' (only mode used in this version)
    dataset_name : string used in saved filenames
    subject_ids  : np.ndarray — subject-level split if provided,
                   random 80/20 split if None
    """
    # ── Remove NaN rows ────────────────────────────────────────────────────
    valid = ~np.any(np.isnan(X), axis=1)
    X     = X[valid]
    y     = y[valid]
    if subject_ids is not None:
        subject_ids = subject_ids[valid]

    print(f"[{dataset_name.upper()}] Samples after NaN removal: {len(X)}")

    if len(X) < 50:
        print(f"[{dataset_name.upper()}] Not enough samples. Aborting.")
        return False

    # ── Train/test split ───────────────────────────────────────────────────
    if subject_ids is not None:
        # Subject-level split — no subject appears in both train and test
        # Critical for correct accuracy evaluation — prevents data leakage
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

        print(f"[{dataset_name.upper()}] Subject-level split "
              f"(prevents data leakage):")
        print(f"  Train: {len(X_tr)} samples "
              f"({len(train_subjects)} subjects)")
        print(f"  Test:  {len(X_te)} samples "
              f"({len(test_subjects)} subjects)")
    else:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        print(f"[{dataset_name.upper()}] Random split: "
              f"Train={len(X_tr)}  Test={len(X_te)}")

    # ── Normalize ──────────────────────────────────────────────────────────
    scaler_X   = StandardScaler()
    scaler_SBP = StandardScaler()
    scaler_DBP = StandardScaler()

    X_tr_sc   = scaler_X.fit_transform(X_tr)
    X_te_sc   = scaler_X.transform(X_te)

    # Scale targets for training
    sbp_tr_sc = scaler_SBP.fit_transform(y_tr[:, 0:1])
    dbp_tr_sc = scaler_DBP.fit_transform(y_tr[:, 1:2])
    y_tr_sc   = np.hstack([sbp_tr_sc, dbp_tr_sc])

    # ── Train Random Forest ────────────────────────────────────────────────
    # Random Forest chosen over XGBoost for this dataset because:
    # - Only 654 samples → RF handles small datasets better
    # - XGBoost's regularization over-penalizes weak signals at this scale
    # - RF's parallel averaging is more stable than sequential boosting
    #   when test subjects are completely unseen (subject-level split)
    print(f"[{dataset_name.upper()}] Training Random Forest (500 trees)...")
    xgb = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=2,
            n_jobs=-1,
            verbosity=0
        )
    )
    xgb.fit(X_tr_sc, y_tr_sc)

    # ── Evaluate ───────────────────────────────────────────────────────────
    # y_pred is in normalized space → needs inverse transform
    # y_te is already in mmHg → do NOT inverse transform
    y_pred   = xgb.predict(X_te_sc)
    sbp_true = y_te[:, 0].flatten()
    sbp_pred = scaler_SBP.inverse_transform(y_pred[:, 0:1]).flatten()
    dbp_true = y_te[:, 1].flatten()
    dbp_pred = scaler_DBP.inverse_transform(y_pred[:, 1:2]).flatten()

    mae_sbp = mean_absolute_error(sbp_true, sbp_pred)
    mae_dbp = mean_absolute_error(dbp_true, dbp_pred)
    std_sbp = np.std(sbp_true - sbp_pred)
    std_dbp = np.std(dbp_true - dbp_pred)

    print(f"\n[{dataset_name.upper()}] Results:")
    print(f"  SBP  MAE={mae_sbp:.2f} mmHg  STD={std_sbp:.2f} mmHg"
          f"  AAMI={'PASS' if mae_sbp <= 5 and std_sbp <= 8 else 'FAIL'}")
    print(f"  DBP  MAE={mae_dbp:.2f} mmHg  STD={std_dbp:.2f} mmHg"
          f"  AAMI={'PASS' if mae_dbp <= 5 and std_dbp <= 8 else 'FAIL'}")

    # ── Save ───────────────────────────────────────────────────────────────
    # Save with suffix for traceability
    suffix = f"{dataset_name}_{mode}"
    joblib.dump(xgb,           f'models/xgb_bp_model_{suffix}.pkl')
    joblib.dump(scaler_X,     f'models/scaler_X_{suffix}.pkl')
    joblib.dump(scaler_SBP,   'models/scaler_SBP.pkl')
    joblib.dump(scaler_DBP,   'models/scaler_DBP.pkl')
    joblib.dump(FEATURE_COLS, f'models/bp_feature_cols_{suffix}.pkl')

    # Also save with standard names that inference.py loads by default
    if mode == 'ppg_only':
        joblib.dump(xgb,           'models/xgb_bp_model_ppg.pkl')
        joblib.dump(scaler_X,     'models/scaler_X_ppg.pkl')
        joblib.dump(FEATURE_COLS, 'models/bp_feature_cols_ppg.pkl')

    print(f"[{dataset_name.upper()}] Saved to models/")
    return True


# =============================================================================
#  BLOOD PRESSURE MODEL — PPG-BP Figshare Dataset
#  PPG-only version — no ECG used or required
# =============================================================================

def train_bp_model():
    """
    Train PPG-only BP model using PPG-BP Figshare dataset.

    Why this dataset:
      1. Raw waveforms — we extract rich APG features (aging_index,
         reflection_index, apg_a, apg_b) not available in pre-computed CSVs
      2. Direct cuff BP labels — matches wearable measurement use case
      3. Subject IDs preserved — proper patient-level split, no data leakage
      4. Diverse subjects — ages 20-89, covers hypertension and diabetes
    """
    from scipy.signal import resample
    from signal_processing import (
        preprocess_ppg, segment_ppg_beats, extract_bp_features
    )
    from tqdm import tqdm

    print("\n" + "=" * 60)
    print("TRAINING BLOOD PRESSURE MODEL")
    print("Dataset: PPG-BP Figshare (Liang et al. 2018)")
    print("Mode: PPG-only")
    print("=" * 60)

    data_dir = 'data/ppg_bp'
    subj_dir = os.path.join(data_dir, '0_subject')

    # ── Validate paths ─────────────────────────────────────────────────────
    if not os.path.exists(data_dir):
        print(f"[PPGBP] Folder not found: {data_dir}")
        print("[PPGBP] Download from:")
        print("[PPGBP]   https://figshare.com/articles/dataset/"
              "PPG-BP_Database_zip/5459299")
        print("[PPGBP] Extract to: data/ppg_bp/")
        return False

    if not os.path.exists(subj_dir):
        print(f"[PPGBP] Subject folder not found: {subj_dir}")
        print("[PPGBP] Expected: data/ppg_bp/0_subject/")
        return False

    # ── Load labels from PPG-BP.xlsx ───────────────────────────────────────
    labels = _load_ppg_bp_labels(data_dir)
    if labels is None:
        return False

    # ── Discover PPG files ─────────────────────────────────────────────────
    # Files are named {subject_id}_{segment_num}.txt in a flat folder
    all_txt_files = glob.glob(os.path.join(subj_dir, '*.txt'))
    print(f"[PPGBP] Found {len(all_txt_files)} PPG text files in 0_subject/")

    if len(all_txt_files) == 0:
        print("[PPGBP] No .txt files found in 0_subject/")
        print("[PPGBP] Expected: 2_1.txt, 2_2.txt, 3_1.txt, ...")
        return False

    # Build lookup: subject_id → list of (segment_num, filepath)
    subject_files = {}
    for fpath in all_txt_files:
        fname = os.path.basename(fpath)
        name  = os.path.splitext(fname)[0]       # e.g. "2_1"
        parts = name.rsplit('_', 1)              # split on last underscore
        if len(parts) != 2:
            continue
        try:
            subj_id = int(parts[0])
            seg_num = int(parts[1])
        except ValueError:
            continue
        if subj_id not in subject_files:
            subject_files[subj_id] = []
        subject_files[subj_id].append((seg_num, fpath))

    for sid in subject_files:
        subject_files[sid].sort(key=lambda x: x[0])

    print(f"[PPGBP] Subjects with PPG files: {len(subject_files)}")
    print(f"[PPGBP] Subject ID range in files: "
          f"{min(subject_files.keys())} – {max(subject_files.keys())}")

    # ── Feature columns ────────────────────────────────────────────────────
    # These must match exactly what extract_bp_features() returns
    # in signal_processing.py — order matters for inference
    FEATURE_COLS = [
        'HR',               # heart rate from PPG peak spacing
        'IH',               # systolic amplitude
        'IL',               # diastolic amplitude
        'PIR',              # PPG intensity ratio (IH / IL)
        'Meu',              # mean PPG amplitude
        'ppg_std',          # standard deviation
        'rise_time',        # onset to systolic peak (replaces PTT)
        'pulse_width',      # width at 50% amplitude
        'apg_a',            # APG a-wave
        'apg_b',            # APG b-wave
        'aging_index',      # b/a ratio — best PPG-only arterial stiffness proxy
        'reflection_index', # post-systolic area fraction
        'auc',              # area under PPG curve
        'max_slope',        # maximum upstroke slope
        'ppg_skew',         # waveform skewness
        'ppg_kurt',         # waveform kurtosis
    ]

    FS_ORIGINAL = 1000   # PPG-BP native sampling rate
    FS_TARGET   = 125    # ESP32 sampling rate

    X_list      = []
    y_list      = []
    subject_ids = []
    n_skipped   = 0

    print("[PPGBP] Extracting features from PPG waveforms...")

    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        subj_id = int(row['subject_id'])
        sbp     = float(row['SBP'])
        dbp     = float(row['DBP'])

        # Physiological outlier filter
        if not (75 <= sbp <= 200 and 40 <= dbp <= 130 and sbp > dbp):
            n_skipped += 1
            continue

        if subj_id not in subject_files:
            n_skipped += 1
            continue

        for seg_num, seg_path in subject_files[subj_id]:
            try:
                # Load raw PPG (2100 samples at 1000 Hz = 2.1 seconds)
                ppg_raw = np.loadtxt(seg_path).flatten().astype(np.float32)

                if len(ppg_raw) < 500:
                    continue

                # Resample 1000 Hz → 125 Hz to match ESP32
                n_target = int(len(ppg_raw) * FS_TARGET / FS_ORIGINAL)
                ppg_125  = resample(ppg_raw, n_target).astype(np.float32)

                if len(ppg_125) < FS_TARGET:
                    continue

                # Preprocessing pipeline — identical to inference.py
                ppg = preprocess_ppg(ppg_125, fs=FS_TARGET)

                # Segment into beat windows
                beats = segment_ppg_beats(ppg, fs=FS_TARGET, window_sec=1.0)
                if len(beats) == 0:
                    continue

                # Use middle beat — least affected by edge artifacts
                ppg_seg = beats[len(beats) // 2]

                # Extract 16 BP features
                feats = extract_bp_features(ppg_seg, fs=FS_TARGET)
                if feats is None:
                    continue

                X_list.append([feats.get(col, 0.0) for col in FEATURE_COLS])
                y_list.append([sbp, dbp])
                subject_ids.append(subj_id)

            except Exception:
                continue

    print(f"\n[PPGBP] Subjects skipped (outlier/missing): {n_skipped}")
    print(f"[PPGBP] Valid segments extracted: {len(X_list)}")

    if len(X_list) == 0:
        print("[PPGBP] No valid segments extracted.")
        print("[PPGBP] Check that subject IDs in PPG-BP.xlsx match "
              "filenames in 0_subject/")
        return False

    X           = np.array(X_list,      dtype=np.float32)
    y           = np.array(y_list,      dtype=np.float32)
    subject_ids = np.array(subject_ids, dtype=np.int32)

    print(f"[PPGBP] Unique subjects with valid data: "
          f"{len(np.unique(subject_ids))}")
    print(f"[PPGBP] SBP: min={y[:,0].min():.1f}  "
          f"max={y[:,0].max():.1f}  mean={y[:,0].mean():.1f} mmHg")
    print(f"[PPGBP] DBP: min={y[:,1].min():.1f}  "
          f"max={y[:,1].max():.1f}  mean={y[:,1].mean():.1f} mmHg")

    return _train_and_save_bp(
        X, y, FEATURE_COLS,
        mode='ppg_only',
        dataset_name='ppg_bp',
        subject_ids=subject_ids
    )


# =============================================================================
#  BLOOD GLUCOSE MODEL — BIG IDEAs Lab Glycemic Variability Dataset
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
        print("[BIGIDEAS] Download from:")
        print("[BIGIDEAS]   https://physionet.org/content/"
              "big-ideas-glycemic-wearable/1.1.2/")
        print("[BIGIDEAS] Requires free PhysioNet account.")
        print("[BIGIDEAS] After registration, download with:")
        print("[BIGIDEAS]   wget -r -N -c -np \\")
        print("[BIGIDEAS]     https://physionet.org/files/"
              "big-ideas-glycemic-wearable/1.1.2/ \\")
        print("[BIGIDEAS]     -P data/bigideas/")
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
        print("[BIGIDEAS] Expected: 001, 002, ..., 016")
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
    gb = XGBRegressor(n_estimators=300, learning_rate=0.05,
                  max_depth=5, subsample=0.8, random_state=42,
                  verbosity=0)
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
    print("BP Dataset:      PPG-BP Figshare (Liang et al. 2018)")
    print("Glucose Dataset: BIG IDEAs Lab   (Bent et al. 2021)")
    print("=" * 60)

    bp_ok  = train_bp_model()
    glu_ok = train_glucose_model()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  BP model:      "
          f"{'OK' if bp_ok  else 'SKIPPED — dataset missing or error'}")
    print(f"  Glucose model: "
          f"{'OK' if glu_ok else 'SKIPPED — dataset missing or error'}")
    print("=" * 60)

    if bp_ok and glu_ok:
        print("\nAll models trained. Run: python server.py")

    if not bp_ok:
        print("\nBP model not trained:")
        print("  1. Download PPG-BP.xlsx from figshare")
        print("  2. Extract to: data/ppg_bp/")
        print("  3. Verify: data/ppg_bp/PPG-BP.xlsx")
        print("  4. Verify: data/ppg_bp/0_subject/2_1.txt")

    if not glu_ok:
        print("\nGlucose model not trained:")
        print("  1. Register at https://physionet.org/register/")
        print("  2. Download big-ideas-glycemic-wearable/1.1.2/")
        print("  3. Extract to: data/bigideas/")
        print("  4. Verify: data/bigideas/001/BVP.csv")
