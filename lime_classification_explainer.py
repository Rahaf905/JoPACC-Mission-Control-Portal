import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import (LimeTabularExplainer)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "pkl_files")

_data = joblib.load(os.path.join(PKL_DIR, "preprocessed_classification.pkl"))


X_train = _data["X_train"]
_activity_encoder = _data["activity_encoder"]
_model = joblib.load(os.path.join(PKL_DIR, "best_classification_model.pkl"))

X_train_df = pd.DataFrame(X_train,columns=X_train.columns)

class_names = [
    _activity_encoder.inverse_transform([int(class_number)])[0]
    for class_number in _model.classes_
]

explainer = LimeTabularExplainer(
    training_data=X_train_df.values,
    feature_names=X_train_df.columns.tolist(),
    class_names=class_names,
    mode="classification",
    random_state=43
)

def explain_using_lime_classification(input_values):

    input_values = np.array(input_values).flatten()
    sample_df = pd.DataFrame([input_values],columns=X_train_df.columns)

    predicted_class = _model.predict(sample_df)[0]
    predicted_probabilities = _model.predict_proba(sample_df)[0]
    class_index = np.where(_model.classes_ == predicted_class)[0][0]
    class_index = int(class_index)
    exp = explainer.explain_instance(
        sample_df.values[0],
        lambda x: _model.predict_proba(pd.DataFrame(x,columns=X_train_df.columns)),
        labels=[class_index],
        num_features=len(X_train_df.columns)
    )

    print("Predicted class:", predicted_class)
    print("Predicted probability:",predicted_probabilities[class_index])
    print("Explanation for sample:")

    explanation_list = exp.as_list(label=class_index)

    explanation_df = pd.DataFrame(explanation_list,columns=["Feature", "Contribution"])

    return explanation_df