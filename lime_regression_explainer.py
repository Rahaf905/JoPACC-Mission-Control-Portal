import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "pkl_files")


_data = joblib.load(os.path.join(PKL_DIR,"preprocessed_regression.pkl"))
X_train = _data["X_train"]
_model = joblib.load(os.path.join(PKL_DIR,"best_regression_model.pkl"))


X_train_df = pd.DataFrame(X_train,columns=X_train.columns)


explainer = LimeTabularExplainer(training_data=X_train_df.values,feature_names=X_train_df.columns.tolist(),mode="regression",random_state=43)


def explain_using_lime(input_values):
    input_values = np.array(input_values).flatten()
    sample_df = pd.DataFrame([input_values],columns=X_train_df.columns)

    prediction = float(_model.predict(sample_df)[0])
    exp = explainer.explain_instance(sample_df.values[0],lambda x: _model.predict(
            pd.DataFrame(
                x,
                columns=X_train_df.columns
            )
        )
    )

    print("prediction: ", prediction)
    print("Explanation for sample:")

    explanation_list_reg = exp.as_list()

    explanation_df = pd.DataFrame(explanation_list_reg,columns=['Feature', 'Contribution'])

    return explanation_df