import pandas as pd
import numpy as np
import skfuzzy as fuzz

df = pd.read_csv("jopacc_system_month_november_2022_to_june_2026.csv")
jomopay = df[df["system"] == "JoMoPay"].copy()
jomopay["report_month"] = pd.to_datetime(jomopay["report_month"])
jomopay = jomopay.sort_values("report_month").reset_index(drop=True) # Sort data before using shift(1)

jomopay["transaction_growth"] = ((jomopay["transactions_total"] - jomopay["transactions_total"].shift(1))/ jomopay["transactions_total"].shift(1)) * 100
jomopay["user_growth"] = ((jomopay["users_total"] - jomopay["users_total"].shift(1)) / jomopay["users_total"].shift(1)) * 100
jomopay["purchase_share"] = (jomopay["tx_purchases_count"] / jomopay["transactions_total"]) * 100
jomopay["cashout_dependency"] = (jomopay["tx_cash_out_count"] / jomopay["transactions_total"]) * 100

transaction_growth_universe = np.arange(-15,31,1)
user_growth_universe = np.arange(-10,11,1)
purchase_share_universe = np.arange(0,21,1)
cashout_universe = np.arange(0,11,1)
growth_quality_universe = np.arange(0, 101, 1)
# =========================
# Input Membership Functions
# =========================
# Transaction Growth
transaction_low = fuzz.trimf(transaction_growth_universe,[-15, -15, 5.314])
transaction_medium = fuzz.trimf(transaction_growth_universe,[-0.628, 5.314, 12.717])
transaction_high = fuzz.trimf(transaction_growth_universe,[5.314, 30, 30])

# User Growth
user_low = fuzz.trimf(user_growth_universe,[-10, -10, 1.266])
user_medium = fuzz.trimf(user_growth_universe,[0.563, 1.266, 1.494])
user_high = fuzz.trimf(user_growth_universe,[1.266, 10, 10])

# Purchase Share
purchase_low = fuzz.trimf(purchase_share_universe,[0, 0, 4.106])
purchase_medium = fuzz.trimf(purchase_share_universe,[3.405, 4.106, 7.827])
purchase_high = fuzz.trimf(purchase_share_universe,[4.106, 20, 20])

# Cash-out Dependency
cashout_low = fuzz.trimf(cashout_universe,[0, 0, 3.419])
cashout_medium = fuzz.trimf(cashout_universe,[1.921, 3.419, 5.261])
cashout_high = fuzz.trimf(cashout_universe,[3.419, 10, 10])
# =========================
# Output Membership Functions
# =========================
growth_fragile = fuzz.trimf(growth_quality_universe,[0, 0, 30])
growth_unbalanced = fuzz.trimf(growth_quality_universe,[20, 40, 60])
growth_healthy = fuzz.trimf(growth_quality_universe,[50, 70, 85])
growth_sustainable = fuzz.trimf(growth_quality_universe,[75, 100, 100])

def fuzzify(value,universe,membership_function):
    return fuzz.interp_membership(universe,membership_function,value)

def apply_rules(transaction_growth,user_growth,purchase_share,cashout_dependency):
    # Fuzzify Transaction Growth
    transaction_low_value = fuzzify(transaction_growth,transaction_growth_universe,transaction_low)
    transaction_medium_value = fuzzify(transaction_growth,transaction_growth_universe,transaction_medium)
    transaction_high_value = fuzzify(transaction_growth,transaction_growth_universe,transaction_high)

    # Fuzzify User Growth
    user_low_value = fuzzify(user_growth,user_growth_universe,user_low)
    user_medium_value = fuzzify(user_growth,user_growth_universe,user_medium)
    user_high_value = fuzzify(user_growth,user_growth_universe,user_high)

    # Fuzzify Purchase Share
    purchase_low_value = fuzzify(purchase_share,purchase_share_universe,purchase_low)
    purchase_medium_value = fuzzify(purchase_share,purchase_share_universe,purchase_medium)
    purchase_high_value = fuzzify(purchase_share,purchase_share_universe,purchase_high)

    # Fuzzify Cash-out Dependency
    cashout_low_value = fuzzify(cashout_dependency,cashout_universe,cashout_low)
    cashout_medium_value = fuzzify(cashout_dependency,cashout_universe,cashout_medium)
    cashout_high_value = fuzzify(cashout_dependency,cashout_universe,cashout_high)

    rule1 = transaction_low_value
    rule2 = np.fmin(user_low_value,cashout_high_value)
    rule3 = np.fmin(purchase_low_value,cashout_high_value)
    rule4 = np.fmin(transaction_medium_value,user_low_value)
    rule5 = np.fmin(transaction_medium_value,purchase_low_value)
    rule6 = np.fmin(transaction_medium_value,cashout_high_value)
    rule7 = np.fmin(transaction_high_value,user_low_value)
    rule8 = np.fmin(transaction_high_value,purchase_low_value)
    rule9 = np.fmin(transaction_high_value,cashout_high_value)
    rule10 = np.fmin(transaction_medium_value,np.fmin(user_medium_value,cashout_low_value))
    rule11 = np.fmin(transaction_high_value,np.fmin(user_medium_value,cashout_medium_value))
    rule12 = np.fmin(transaction_high_value,np.fmin(user_high_value,np.fmin(purchase_high_value,cashout_low_value)))
    # =========================
    # Combine Rules by Output
    # =========================
    fragile_activation = np.fmax(rule1,np.fmax(rule2, rule3))
    unbalanced_activation = np.fmax(rule4,np.fmax(rule5,np.fmax(rule6,np.fmax(rule7,np.fmax(rule8, rule9)))))
    healthy_activation = np.fmax(rule10,rule11)
    sustainable_activation = rule12
    
    rule_activations = {
        "Rule 1: Low transaction growth → Fragile": rule1,
        "Rule 2: Low user growth and high cash-out dependency → Fragile": rule2,
        "Rule 3: Low purchase share and high cash-out dependency → Fragile": rule3,
        "Rule 4: Medium transaction growth and low user growth → Unbalanced": rule4,
        "Rule 5: Medium transaction growth and low purchase share → Unbalanced": rule5,
        "Rule 6: Medium transaction growth and high cash-out dependency → Unbalanced": rule6,
        "Rule 7: High transaction growth and low user growth → Unbalanced": rule7,
        "Rule 8: High transaction growth and low purchase share → Unbalanced": rule8,
        "Rule 9: High transaction growth and high cash-out dependency → Unbalanced": rule9,
        "Rule 10: Medium transaction growth, medium user growth and low cash-out dependency → Healthy": rule10,
        "Rule 11: High transaction growth, medium user growth and medium cash-out dependency → Healthy": rule11,
        "Rule 12: High transaction growth, high user growth, high purchase share and low cash-out dependency → Sustainable": rule12
}
    return (
        fragile_activation,
        unbalanced_activation,
        healthy_activation,
        sustainable_activation,
        rule_activations
    )
# =========================
# Aggregate Fuzzy Outputs
# =========================
def aggregate(fragile,unbalanced,healthy,sustainable):
    fragile_cut = np.fmin(fragile,growth_fragile)
    unbalanced_cut = np.fmin(unbalanced,growth_unbalanced)
    healthy_cut = np.fmin(healthy,growth_healthy)
    sustainable_cut = np.fmin(sustainable,growth_sustainable)
    aggregated = np.fmax(fragile_cut,np.fmax(unbalanced_cut,np.fmax(healthy_cut,sustainable_cut)))

    return aggregated
# =========================
# Centroid Defuzzification
# =========================
def centroid_defuzz(aggregated):
    if np.max(aggregated) == 0:
        return 0.0
    
    return fuzz.defuzz(growth_quality_universe,aggregated,"centroid")
# =========================
# Determine Growth Level
# =========================
def get_growth_level(score):
    memberships = {
        "Fragile Growth": fuzz.interp_membership(growth_quality_universe,growth_fragile,score),
        "Unbalanced Growth": fuzz.interp_membership(growth_quality_universe,growth_unbalanced,score),
        "Healthy Growth": fuzz.interp_membership(growth_quality_universe,growth_healthy,score),
        "Sustainable Growth": fuzz.interp_membership(growth_quality_universe,growth_sustainable,score)
    }

    return max(memberships, key=memberships.get)
# =========================
# Get Activated Rules
# =========================
def get_activated_rules(rule_activations):
    activated_rules = []

    for rule_name, activation_value in rule_activations.items():
        if activation_value > 0:
            activated_rules.append(
                {
                    "rule": rule_name,
                    "activation": round(float(activation_value),3)
                }
            )
    activated_rules.sort(key=lambda item: item["activation"],reverse=True)

    return activated_rules
# =========================
# Generate Explanation
# =========================
def generate_explanation(level,activated_rules,transaction_growth,user_growth,purchase_share,cashout_dependency):
    input_summary = (
        f"Transaction growth is {transaction_growth:.2f}%, "
        f"user growth is {user_growth:.2f}%, "
        f"purchase share is {purchase_share:.2f}% "
        f"and cash-out dependency is {cashout_dependency:.2f}%."
    )
    if not activated_rules:
        return ("No fuzzy rule was activated for the entered values. " + input_summary)

    strongest_rule = activated_rules[0]
    return (
        f"The result is {level}. "
        f"The strongest activated rule was: "
        f"{strongest_rule['rule']}. "
        + input_summary
    ) 

def calculate_growth_quality(transaction_growth,user_growth,purchase_share,cashout_dependency):
    # Keep inputs inside universe ranges
    transaction_growth = float(np.clip(transaction_growth,transaction_growth_universe.min(),transaction_growth_universe.max()))
    user_growth = float(np.clip(user_growth,user_growth_universe.min(),user_growth_universe.max()))
    purchase_share = float(np.clip(purchase_share,purchase_share_universe.min(),purchase_share_universe.max()))
    cashout_dependency = float(np.clip(cashout_dependency,cashout_universe.min(),cashout_universe.max()))

    # Apply fuzzy rules
    (fragile,unbalanced,healthy,sustainable,rule_activations) = apply_rules(transaction_growth,user_growth,purchase_share,cashout_dependency)
    # Aggregate fuzzy outputs
    aggregated = aggregate(fragile,unbalanced,healthy,sustainable)
    #  Growth Quality Score
    growth_score = round(centroid_defuzz(aggregated),2)
    growth_level = get_growth_level(growth_score)
    activated_rules = get_activated_rules(rule_activations)
    explanation = generate_explanation(growth_level, activated_rules,transaction_growth,user_growth,purchase_share,cashout_dependency)
    return (growth_score,growth_level,activated_rules,explanation)