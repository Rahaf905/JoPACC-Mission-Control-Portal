import os
from google import genai


GEMINI_API_KEY = "Your API KEY"
client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini(prompt):
    """Send a prompt to Gemini and return the text response."""

    response = client.models.generate_content(model="gemini-3-flash-preview",contents=prompt)
    return response.text

def explain_regression_with_gemini(prediction,shap_table,lime_table):
    """
    Explain the regression prediction using SHAP and LIME in simple language.
    """
    prompt = f""" You are a helpful teaching assistant explaining a machine learning prediction 
for Jordan's payment systems.

The regression model predicted the following value:
{prediction}

Here are the SHAP values showing how much each feature
increased or decreased the predicted value:
{shap_table}

Here are the LIME contributions showing the local effect
of each feature:
{lime_table}

Please explain the result in 3-4 simple sentences:
1. Explain what the predicted value means.
2. Identify the features with the largest positive and
   negative effects according to SHAP.
3. Explain whether LIME generally agrees with SHAP.
"""
    return ask_gemini(prompt)

def explain_classification_with_gemini(predicted_class,probability,shap_table,lime_table):
    """
    Explain the classification prediction using SHAP and LIME.
    """
    prompt = f"""You are a helpful teaching assistant explaining a machine
learning classification result for Jordan's payment systems.

The classification model predicted the following class:
{predicted_class}

The probability of the predicted class is:
{probability}

Here are the SHAP values showing how each feature affected
the predicted class:
{shap_table}

Here are the LIME contributions showing the local effect
of each feature:
{lime_table}

Please explain the result in 3-4 simple sentences:

1. Explain what the predicted class and its probability mean.
2. Identify the features with the largest positive and
   negative effects according to SHAP.
3. Explain whether LIME generally agrees with SHAP.
"""
    return ask_gemini(prompt)