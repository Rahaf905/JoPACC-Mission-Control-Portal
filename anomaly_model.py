import os
import pickle
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "pkl_files")

_anomaly_model = pickle.load(open(os.path.join(PKL_DIR,"best_anomaly_model.pkl"),"rb"))
_anomaly_threshold = pickle.load(open(os.path.join(PKL_DIR,"anomaly_threshold.pkl"),"rb"))
_ranked_anomalies = pd.read_pickle(os.path.join(PKL_DIR, "ranked_anomalies.pkl"))
_anomaly_scaler = pickle.load(open(os.path.join(PKL_DIR, "scaler.pkl"),"rb"))
_anomaly_ohe = pickle.load(open(os.path.join(PKL_DIR,"onehot_encoder.pkl"),"rb"))
_num_cols = pickle.load(open(os.path.join(PKL_DIR, "num_cols.pkl"),"rb"))
_cat_cols = pickle.load(open(os.path.join(PKL_DIR, "cat_cols.pkl"),"rb"))
_system_reference_stats = pickle.load(open(os.path.join(PKL_DIR,"system_reference_stats.pkl"),"rb"))

def check_anomaly(input_values, system):

    numeric_values = [float(input_values[feature])for feature in _num_cols]

    numeric_df = pd.DataFrame([numeric_values],columns=_num_cols)
    X_num = _anomaly_scaler.transform(numeric_df)
    categorical_df = pd.DataFrame([[system]],columns=_cat_cols)
    X_cat = _anomaly_ohe.transform(categorical_df)
    X_input = np.hstack([X_num,X_cat])
    anomaly_score = float(-_anomaly_model.decision_function(X_input)[0])
    is_anomalous = bool(anomaly_score >= _anomaly_threshold)
    anomaly_label = ("ANOMALOUS" if is_anomalous else "NORMAL")
    system_stats = (_system_reference_stats[system])
    feature_analysis = []

    for feature in _num_cols:
        entered_value = float(
            input_values[feature]
        )

        feature_stats = system_stats[feature]

        feature_mean = feature_stats["mean"]
        feature_std = feature_stats["std"]
        feature_min = feature_stats["min"]
        feature_max = feature_stats["max"]

        if feature_std > 0:
            deviation = abs(entered_value - feature_mean) / feature_std

        elif entered_value == feature_mean:\
            deviation = 0.0

        else:
            deviation = float("inf")


        if entered_value > feature_max:
            reason = ("Higher than the historical range " f"for {system}")
            outside_range = True

        elif entered_value < feature_min:
            reason = (
                "Lower than the historical range " f"for {system}")
            
            outside_range = True

        else:
            reason = (
                "Shows a large deviation from the " f"typical value for {system}")

            outside_range = False


        feature_analysis.append({
            "feature": feature,
            "entered_value": entered_value,
            "minimum": feature_min,
            "maximum": feature_max,
            "deviation": deviation,
            "reason": reason,
            "outside_range": outside_range
        })

    unusual_features = [
        feature
        for feature in feature_analysis
        if feature["outside_range"]
    ]

    if is_anomalous and not unusual_features:
        unusual_features = sorted(feature_analysis,
            key=lambda feature: feature["deviation"],reverse=True)[:3]

        explanation = (
            "No single input value was outside the "
            "historical range. However, the combination "
            "of values was unusual. The listed features "
            "showed the largest deviations from the "
            f"typical values for {system}."
        )

    elif is_anomalous:
        unusual_features = sorted(
            unusual_features,
            key=lambda feature: feature["deviation"],reverse=True)[:5]
        explanation = (
            "The input was classified as anomalous "
            "because one or more values were outside "
            f"the historical operating range for {system}. "
            "The prediction and XAI processes were stopped."
        )

    else:
        unusual_features = []
        explanation = (
            "The entered values are within the reliable "
            "operating range. Prediction and explanation "
            "processing may continue."
        )

    for feature in unusual_features:
        if not np.isfinite(feature["deviation"]):
            feature["deviation"] = None

    return {
        "is_anomalous": is_anomalous,
        "label": anomaly_label,
        "score": round(anomaly_score, 4),
        "threshold": round(float(_anomaly_threshold),4),
        "unusual_features": unusual_features,
        "explanation": explanation
    }

def get_ranked_anomalies(limit=10):
    return _ranked_anomalies.head(limit).to_dict("records")