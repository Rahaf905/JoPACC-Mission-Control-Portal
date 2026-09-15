import joblib
import numpy as np
import pandas as pd
import os
from lime_regression_explainer import explain_using_lime
from lime_classification_explainer import explain_using_lime_classification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "pkl_files")

_system_encoder = joblib.load(os.path.join(PKL_DIR, "system_encoder.pkl"))
_regression_model = joblib.load(os.path.join(PKL_DIR, "best_regression_model.pkl"))
_classification_model = joblib.load(os.path.join(PKL_DIR, "best_classification_model.pkl"))
_regression_preprocessing = joblib.load(os.path.join(PKL_DIR, "preprocessed_regression.pkl"))
_classification_preprocessing = joblib.load(os.path.join(PKL_DIR, "preprocessed_classification.pkl"))
_regression_scaler = _regression_preprocessing["scaler"]
_classification_scaler = ( _classification_preprocessing["scaler"])
_activity_encoder = (_classification_preprocessing["activity_encoder"])

REGRESSION_FEATURE_NAMES = (_regression_preprocessing["selected_features"])
CLASSIFICATION_FEATURE_NAMES = (_classification_preprocessing["selected_features"])
REGRESSION_NUMERIC_FEATURE_NAMES = (_regression_preprocessing["numeric_features"])
CLASSIFICATION_NUMERIC_FEATURE_NAMES = (_classification_preprocessing["numeric_features"])

_regression_shap_data = joblib.load(os.path.join(PKL_DIR,"shap_regression_explainer.pkl"))
_regression_shap_explainer = (_regression_shap_data["explainer"])
_classification_shap_data = joblib.load(os.path.join(PKL_DIR,"shap_classification_explainer.pkl"))
_classification_shap_explainer = (_classification_shap_data["explainer"])

def collect_regression_data(
    report_year,
    report_month_number,
    users_total,
    transactions_total,
    customer_legal_entity_count,
    wallet_individual_count,
    user_new_count,
    tx_purchases_count,
    tx_money_transfers_count,
    ef_digital_payment_count,
    ach_jod_transaction_count,
    returned_cheque_count,
    system
):
    system_encoded = _system_encoder.transform(pd.DataFrame({"system": [system]}))[0]
    input_data = [
        report_year,
        report_month_number,
        users_total,
        transactions_total,
        customer_legal_entity_count,
        wallet_individual_count,
        user_new_count,
        tx_purchases_count,
        tx_money_transfers_count,
        ef_digital_payment_count,
        ach_jod_transaction_count,
        returned_cheque_count
    ] + list(system_encoded)

    X = pd.DataFrame([input_data],columns=REGRESSION_FEATURE_NAMES,dtype=float)
    X[REGRESSION_NUMERIC_FEATURE_NAMES] = ( _regression_scaler.transform(X[REGRESSION_NUMERIC_FEATURE_NAMES]))
    return X

def predict_regression(data):
    prediction = _regression_model.predict(data)[0]

    return round(float(prediction), 2)


def explain_regression_shap(data):
    shap_values = (_regression_shap_explainer.shap_values(data)[0])
    shap_df = pd.DataFrame({
        "Feature": REGRESSION_FEATURE_NAMES,
        "Feature Value": data.iloc[0].values,
        "SHAP Value": shap_values.round(4)
    })
    return shap_df.to_dict("records")

def explain_regression_lime(data):
    lime_df = explain_using_lime(data.iloc[0].values)
    lime_df["Contribution"] = (lime_df["Contribution"].round(4))
    return lime_df.to_dict("records")

def collect_classification_data(
    report_year,
    report_month_number,
    users_total,
    transactions_total,
    customer_legal_entity_count,
    wallet_individual_count,
    user_new_count,
    tx_purchases_count,
    tx_money_transfers_count,
    ef_digital_payment_count,
    ach_jod_transaction_count,
    returned_cheque_count,
    system
):
    system_encoded = _system_encoder.transform(pd.DataFrame({"system": [system]}))[0]
    input_data = [
        report_year,
        report_month_number,
        users_total,
        transactions_total,
        customer_legal_entity_count,
        wallet_individual_count,
        user_new_count,
        tx_purchases_count,
        tx_money_transfers_count,
        ef_digital_payment_count,
        ach_jod_transaction_count,
        returned_cheque_count
    ] + list(system_encoded)

    X = pd.DataFrame([input_data],columns=CLASSIFICATION_FEATURE_NAMES, dtype=float)
    X[CLASSIFICATION_NUMERIC_FEATURE_NAMES] = (_classification_scaler.transform(X[CLASSIFICATION_NUMERIC_FEATURE_NAMES]))
    return X

def predict_classification(data):
    predicted_class_encoded = (_classification_model.predict(data)[0])
    probabilities = (_classification_model.predict_proba(data)[0])

    class_index = np.where(_classification_model.classes_ == predicted_class_encoded)[0][0]
    probability = probabilities[class_index]
    predicted_class_name = (
        _activity_encoder.inverse_transform(
            [int(predicted_class_encoded)])[0]
    )
    return (predicted_class_name,round(float(probability), 4))


def explain_classification_shap(data):
    predicted_class = (_classification_model.predict(data)[0])
    class_index = np.where(_classification_model.classes_== predicted_class)[0][0]
    shap_values = (_classification_shap_explainer.shap_values(data))

    if isinstance(shap_values, list):
        sample_shap_values = (shap_values[class_index][0])
    elif shap_values.ndim == 3:
        sample_shap_values = (shap_values[0, :, class_index])
    else:
        sample_shap_values = shap_values[0]

    shap_df = pd.DataFrame({
        "Feature": CLASSIFICATION_FEATURE_NAMES,
        "Feature Value": data.iloc[0].values,
        "SHAP Value": sample_shap_values.round(4)
    })

    return shap_df.to_dict("records")

def explain_classification_lime(data):
    lime_df = explain_using_lime_classification(data.iloc[0].values)
    lime_df["Contribution"] = (lime_df["Contribution"].round(4))

    return lime_df.to_dict("records")