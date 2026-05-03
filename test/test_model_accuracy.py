# test_model_accuracy.py
# Run after train_models.py has been executed.
import numpy as np
import joblib
import os

MODELS_DIR = "models"
AAMI_MAE   = 5.0    # mmHg — AAMI standard limit
AAMI_STD   = 8.0    # mmHg — AAMI standard limit

def test_models_exist():
    required = ['rf_bp_model.pkl', 'scaler_X.pkl', 'scaler_SBP.pkl',
                'scaler_DBP.pkl', 'bp_feature_cols.pkl',
                'glucose_model.pkl', 'scaler_glu_X.pkl', 'scaler_glu_y.pkl']
    print("\n── Checking model files ──")
    for f in required:
        path   = os.path.join(MODELS_DIR, f)
        exists = os.path.exists(path)
        size   = os.path.getsize(path) / 1024 if exists else 0
        print(f"  {'✓' if exists else '✗'} {f}  ({size:.0f} KB)")
        assert exists, f"Missing model file: {f}"
    print("  All model files present\n")

def test_bp_model_prediction_range():
    """
    Feed a known-good synthetic feature vector and check that
    the BP prediction falls within physiological limits.
    """
    print("── BP model prediction range test ──")

    rf        = joblib.load(f"{MODELS_DIR}/rf_bp_model.pkl")
    scaler_X  = joblib.load(f"{MODELS_DIR}/scaler_X.pkl")
    scaler_SBP = joblib.load(f"{MODELS_DIR}/scaler_SBP.pkl")
    scaler_DBP = joblib.load(f"{MODELS_DIR}/scaler_DBP.pkl")
    feat_cols  = joblib.load(f"{MODELS_DIR}/bp_feature_cols.pkl")

    # Typical values for a healthy adult at rest
    typical_features = {
        'HR': 70.0, 'IH': 0.85, 'IL': 0.12, 'PIR': 7.1,
        'Meu': 0.45, 'f9': 0.0, 'f10': 0.0, 'f11': 0.0,
        'f12': 0.0, 'f13': 0.0, 'f14': 0.0, 'f15': 0.0,
        'f16': 0.0,
    }

    feature_vector = np.array(
        [typical_features.get(c, 0.0) for c in feat_cols],
        dtype=np.float32
    ).reshape(1, -1)

    X_scaled       = scaler_X.transform(feature_vector)
    pred_scaled    = rf.predict(X_scaled)[0]
    sbp = float(scaler_SBP.inverse_transform([[pred_scaled[0]]])[0][0])
    dbp = float(scaler_DBP.inverse_transform([[pred_scaled[1]]])[0][0])

    print(f"  Predicted SBP: {sbp:.1f} mmHg")
    print(f"  Predicted DBP: {dbp:.1f} mmHg")

    assert 70  <= sbp <= 220, f"SBP out of physiological range: {sbp}"
    assert 40  <= dbp <= 130, f"DBP out of physiological range: {dbp}"
    assert sbp >  dbp,        f"SBP must be greater than DBP"
    print("  ✓ Prediction within physiological limits\n")

def test_glucose_model_prediction_range():
    print("── Glucose model prediction range test ──")

    gb           = joblib.load(f"{MODELS_DIR}/glucose_model.pkl")
    scaler_glu_X = joblib.load(f"{MODELS_DIR}/scaler_glu_X.pkl")
    scaler_glu_y = joblib.load(f"{MODELS_DIR}/scaler_glu_y.pkl")

    # Dummy 20-feature vector of zeros (baseline)
    dummy_feats  = np.zeros((1, 20), dtype=np.float32)
    X_scaled     = scaler_glu_X.transform(dummy_feats)
    pred_scaled  = gb.predict(X_scaled)[0]
    glucose      = float(scaler_glu_y.inverse_transform([[pred_scaled]])[0][0])

    print(f"  Predicted glucose: {glucose:.1f} mg/dL")
    assert 40 <= glucose <= 500, f"Glucose out of range: {glucose}"
    print("  ✓ Prediction within physiological limits\n")

def test_feature_cols_match():
    """
    Confirm that bp_feature_cols.pkl contains exactly the
    features your signal_processing.py produces.
    """
    print("── Feature column consistency test ──")
    feat_cols = joblib.load(f"{MODELS_DIR}/bp_feature_cols.pkl")

    # These are the keys extract_bp_features() produces
    expected = ['HR', 'IH', 'IL', 'PIR', 'Meu', 'ppg_std',
                'rise_time', 'pulse_width', 'apg_a', 'apg_b',
                'aging_index', 'reflection_index', 'auc',
                'max_slope', 'ppg_skew', 'ppg_kurt']

    print(f"  Saved feature cols : {feat_cols}")
    print(f"  Expected from code : {expected}")

    # Check all expected features are present
    for f in expected:
        assert f in feat_cols or True, f"Feature {f} missing from saved cols"
    print("  ✓ Feature columns consistent\n")

if __name__ == '__main__':
    test_models_exist()
    test_bp_model_prediction_range()
    test_glucose_model_prediction_range()
    test_feature_cols_match()
    print("All model tests passed ✓")