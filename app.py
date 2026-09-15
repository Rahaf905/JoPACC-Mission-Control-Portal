from flask import Flask, render_template, request
from model import collect_regression_data,predict_regression,explain_regression_shap,explain_regression_lime
from model import collect_classification_data,predict_classification,explain_classification_shap, explain_classification_lime
from gemini_api import explain_regression_with_gemini,explain_classification_with_gemini
from anomaly_model import check_anomaly,get_ranked_anomalies
from rl_model import recommend_interventions, get_system_features
from fuzzy_jomopay import calculate_growth_quality


app = Flask(__name__)
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/regression", methods=["GET", "POST"])
def regression():
    if request.method == "POST":
        report_year = request.form["report_year"]
        report_month_number = request.form["report_month_number"]
        users_total = request.form["users_total"]
        transactions_total = request.form["transactions_total"]
        customer_legal_entity_count = request.form["customer_legal_entity_count"]
        wallet_individual_count = request.form["wallet_individual_count"]
        user_new_count = request.form["user_new_count"]
        tx_purchases_count = request.form["tx_purchases_count"]
        tx_money_transfers_count = request.form["tx_money_transfers_count"]
        ef_digital_payment_count = request.form["ef_digital_payment_count"]
        ach_jod_transaction_count = request.form["ach_jod_transaction_count"]
        returned_cheque_count = request.form["returned_cheque_count"]
        system = request.form["system"]

        input_values = {
            "report_year": report_year,
            "report_month_number": report_month_number,
            "users_total": users_total,
            "transactions_total": transactions_total,
            "customer_legal_entity_count": customer_legal_entity_count,
            "wallet_individual_count":wallet_individual_count,
            "user_new_count": user_new_count,
            "tx_purchases_count": tx_purchases_count,
            "tx_money_transfers_count":tx_money_transfers_count,
            "ef_digital_payment_count":ef_digital_payment_count,
            "ach_jod_transaction_count":ach_jod_transaction_count,
            "returned_cheque_count":returned_cheque_count

            }

        anomaly_result = check_anomaly(input_values,system)

        if anomaly_result["is_anomalous"]:
            ranked_anomalies = get_ranked_anomalies(10)
            return render_template("regression.html",anomaly_result=anomaly_result,ranked_anomalies=ranked_anomalies)

        data = collect_regression_data(
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
        )

        prediction = predict_regression(data)
        shap_results = explain_regression_shap(data)
        lime_results = explain_regression_lime(data)

        try:
            gemini_text = explain_regression_with_gemini(prediction,shap_results,lime_results)

        except Exception as e:
            gemini_text = f"Gemini API error: {e}"

        return render_template(
            "regression.html",
            anomaly_result=anomaly_result,
            prediction=prediction,
            shap_results=shap_results,
            lime_results=lime_results,
            gemini_text=gemini_text
        )
    
    return render_template("regression.html")

@app.route("/classification",methods=["GET", "POST"])
def classification():
    if request.method == "POST":
        report_year = request.form["report_year"]
        report_month_number = request.form["report_month_number"]
        users_total = request.form["users_total"]
        transactions_total = request.form["transactions_total"]
        customer_legal_entity_count = request.form["customer_legal_entity_count"]
        wallet_individual_count = request.form["wallet_individual_count"]
        user_new_count = request.form["user_new_count"]
        tx_purchases_count = request.form["tx_purchases_count"]
        tx_money_transfers_count = request.form["tx_money_transfers_count"]
        ef_digital_payment_count = request.form["ef_digital_payment_count"]
        ach_jod_transaction_count = request.form["ach_jod_transaction_count"]
        returned_cheque_count = request.form["returned_cheque_count"]
        system = request.form["system"]

        input_values = {
            "report_year": report_year,
            "report_month_number": report_month_number,
            "users_total": users_total,
            "transactions_total": transactions_total,
            "customer_legal_entity_count":customer_legal_entity_count,
            "wallet_individual_count":wallet_individual_count,
            "user_new_count": user_new_count,
            "tx_purchases_count": tx_purchases_count,
            "tx_money_transfers_count":tx_money_transfers_count,
            "ef_digital_payment_count":ef_digital_payment_count,
            "ach_jod_transaction_count":ach_jod_transaction_count,
            "returned_cheque_count":returned_cheque_count
            }
        anomaly_result = check_anomaly(input_values,system)
        if anomaly_result["is_anomalous"]:
            ranked_anomalies = get_ranked_anomalies(10)
            return render_template("classification.html",anomaly_result=anomaly_result,ranked_anomalies=ranked_anomalies)

        data = collect_classification_data(
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
        )

        predicted_class, probability = (predict_classification(data))
        shap_results = explain_classification_shap(data)
        lime_results = explain_classification_lime(data)

        try:
            gemini_text = explain_classification_with_gemini(
                predicted_class,
                probability,
                shap_results,
                lime_results
            )

        except Exception as e:
            gemini_text = f"Gemini API error: {e}"

        return render_template(
            "classification.html",
            anomaly_result=anomaly_result,
            predicted_class=predicted_class,
            probability=probability,
            shap_results=shap_results,
            lime_results=lime_results,
            gemini_text=gemini_text
        )
    
    return render_template("classification.html")

@app.route("/rl", methods=["GET", "POST"])
def rl():
    if request.method == "POST":
        system = request.form["system"]
        available_budget = request.form["available_budget"]
        features = get_system_features(system)
        input_values = {}
        for feature in features:
            input_values[feature] = request.form[feature]

        rl_result = recommend_interventions(system,input_values,available_budget)

        return render_template("rl.html",rl_result=rl_result)

    return render_template("rl.html")


@app.route("/fuzzy", methods=["GET", "POST"])
def fuzzy():

    if request.method == "POST":
        transaction_growth = float(request.form["transaction_growth"])
        user_growth = float(request.form["user_growth"])
        purchase_share = float(request.form["purchase_share"])
        cashout_dependency = float(request.form["cashout_dependency"])
        (growth_score,growth_level,activated_rules,explanation) = calculate_growth_quality(
            transaction_growth,user_growth,purchase_share,cashout_dependency)
        return render_template("fuzzy.html",
            growth_score=growth_score,
            growth_level=growth_level,
            activated_rules=activated_rules,
            explanation=explanation
        )

    return render_template("fuzzy.html")

if __name__ == "__main__":
    app.run(debug=True)