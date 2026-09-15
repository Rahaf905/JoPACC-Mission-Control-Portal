import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "pkl_files")

_rl_policy = joblib.load(os.path.join(PKL_DIR,"best_rl_policy.pkl"))
_policy_name = _rl_policy["policy_name"]
_q_tables = _rl_policy["q_tables"]
_system_features = _rl_policy["system_features"]
_system_bins = _rl_policy["system_bins"]
_system_actions = _rl_policy["system_actions"]
_action_lists = _rl_policy["action_lists"]
_positive_features = _rl_policy["positive_features"]
_negative_features = _rl_policy["negative_features"]
_state_to_index = _rl_policy["state_to_index"]
_max_steps = _rl_policy["max_steps"]
_simulation_note = _rl_policy["simulation_note"]
_reward_parameters = _rl_policy["reward_parameters"]
_improvement_weight = _reward_parameters["improvement_weight"]
_cost_penalty = _reward_parameters["cost_penalty"]
_step_penalty = _reward_parameters["step_penalty"]
_ineffective_penalty = _reward_parameters["ineffective_penalty"]

def create_state(system, input_values):
    state_levels = []
    for feature in _system_features[system]:

        value = float(input_values[feature])
        bins = _system_bins[system][feature]
        # Keep the input inside the training range
        value = np.clip(value,bins[0],bins[-1])
        level = int(pd.cut([value],bins=bins,labels=[0, 1, 2],include_lowest=True)[0] )
        state_levels.append(level)

    return tuple(state_levels)

def state_to_labels(system, state):
    level_names = {0: "Low",1: "Medium",2: "High"}

    labelled_state = {}
    features = _system_features[system]
    for index in range(len(features)):
        feature = features[index]
        level = state[index]
        labelled_state[feature] = (level_names[level])

    return labelled_state

def calculate_state_score(system, state):
    score = 0
    # Positive indicators are better when high
    for index in _positive_features[system]:
        score += state[index]
    # Negative indicators are better when low
    for index in _negative_features[system]:
        score += 2 - state[index]

    return score

def is_goal_state(system, state):
    return (calculate_state_score(system, state) == 8)

def apply_action(system, state, action):

    next_state = list(state)
    if action == "Do Nothing":
        return tuple(next_state)

    if system == "JoMoPay":
        if action == "Promote Wallet Usage":
            next_state[0] = min(next_state[0] + 1,2)
            next_state[1] = min(next_state[1] + 1,2)

        elif action == "Encourage Wallet Cash-In":
            next_state[2] = min(next_state[2] + 1,2)

        elif action == "Reduce Cash-Out Dependency":
            next_state[3] = max(next_state[3] - 1,0)

    elif system == "CliQ":
        if action == "Increase Customer Participation":
            next_state[0] = min(next_state[0] + 1,2)
            next_state[1] = min(next_state[1] + 1,2)

        elif action == "Promote CliQ Purchases":
            next_state[2] = min(next_state[2] + 1,2)

        elif action == "Encourage Money Transfers":
            next_state[3] = min(next_state[3] + 1,2)

    elif system == "eFAWATEERcom":
        if action == "Attract New Users":
            next_state[0] = min(next_state[0] + 1,2)

        elif action == "Onboard New Billers":
            next_state[1] = min(next_state[1] + 1,2)

        elif action == "Move Cash Payments to Digital Channels":
            # Reduce cash payments
            next_state[2] = max(next_state[2] - 1,0)
            # Increase digital payments
            next_state[3] = min(next_state[3] + 1,2)

    elif system == "ACH":
        if action == "Promote JOD Transfers":
            next_state[0] = min(next_state[0] + 1,2)

        elif action == "Promote USD Transfers":
            next_state[1] = min(next_state[1] + 1,2)

        elif action == "Promote EUR and GBP Transfers":
            next_state[2] = min(next_state[2] + 1,2)
            next_state[3] = min(next_state[3] + 1,2)

    elif system == "ECCU":
        if action == "Improve Cheque Processing":
            next_state[0] = min(next_state[0] + 1,2)

        elif action == "Reduce Returned-Cheque Count":
            next_state[1] = max(next_state[1] - 1,0)

        elif action == "Reduce Returned-Cheque Risk Rates":
            next_state[2] = max(next_state[2] - 1,0)
            next_state[3] = max(next_state[3] - 1,0)

    return tuple(next_state)

def apply_action_with_budget(system,state,action,budget):

    action_cost = _system_actions[system][action]
    if action_cost > budget:
        return (state,budget,action_cost,False)

    next_state = apply_action(system,state,action)
    remaining_budget = (budget - action_cost)

    return (next_state,remaining_budget,action_cost,True)

def calculate_reward(system, current_state, next_state, action_cost):

    current_score = calculate_state_score(system, current_state)
    next_score = calculate_state_score(system, next_state)

    improvement = next_score - current_score

    # Encourage operational improvement
    reward = improvement * _improvement_weight

    # Penalise intervention cost
    reward -= action_cost * _cost_penalty

    # Penalise unnecessary steps
    reward -= _step_penalty

    # Penalise ineffective actions
    if next_state == current_state:
        reward -= _ineffective_penalty

    return reward

def check_episode_end(system,state,budget,action,step):

    if is_goal_state(system, state):
        return True, "Goal state reached"

    if action == "Do Nothing":
        return True, "No intervention selected"

    intervention_costs = []

    for (action_name,cost) in _system_actions[system].items():
        if action_name != "Do Nothing":
            intervention_costs.append(cost)

    if budget < min(intervention_costs):
        return True, "Insufficient budget"

    if step >= _max_steps:
        return True, "Maximum steps reached"

    return False, "Continue"

def select_action(system, state, budget):

    state_index = _state_to_index[state]

    affordable_action_indices = []

    for action_index, action in enumerate(_action_lists[system]):

        action_cost = _system_actions[system][action]

        if action_cost <= budget:
            affordable_action_indices.append(action_index)

    q_values = _q_tables[system][state_index]
    best_action_index = max(affordable_action_indices,key=lambda index: q_values[index])

    return _action_lists[system][best_action_index]

def recommend_interventions(system,input_values,available_budget):

    available_budget = float(available_budget)

    current_state = create_state(system,input_values)

    initial_state = current_state
    initial_budget = available_budget
    recommendations = []
    ending_reason = "Continue"

    # No intervention is needed
    # when the initial state is already ideal
    if is_goal_state(system, current_state):
        ending_reason = ("Initial state is already the goal state")

        return {
            "policy_name":_policy_name,
            "system":system,
            "initial_state":state_to_labels(system,initial_state),
            "initial_score": calculate_state_score(system,initial_state),
            "initial_budget":round(float(initial_budget),2),
            "recommendations":recommendations,
            "final_state":state_to_labels(system,current_state),
            "final_score":calculate_state_score(system,current_state),
            "remaining_budget":round(float(available_budget),2),
            "ending_reason":ending_reason,
            "simulation_note":_simulation_note
        }

    for step in range(1, _max_steps + 1):

        action = select_action(system,current_state, available_budget)

        next_state, remaining_budget, action_cost, _ = apply_action_with_budget(
            system, current_state, action, available_budget)
        
        reward = calculate_reward(system,current_state,next_state,action_cost)

        recommendations.append({
            "step":step,
            "action":action,
            "current_state":state_to_labels(system,current_state),
            "next_state":state_to_labels(system,next_state),
            "reward":round(float(reward),2),
            "intervention_cost":round(float(action_cost),2),
            "remaining_budget":round(float(remaining_budget), 2)
        })

        done, ending_reason = check_episode_end(system,next_state,remaining_budget,action,step)
        current_state = next_state
        available_budget = remaining_budget

        if done:
            break

    return {

        "policy_name":_policy_name,
        "system":system,
        "initial_state":state_to_labels(system,initial_state),
        "initial_score":calculate_state_score(system,initial_state),
        "initial_budget":round(float(initial_budget),2),
        "recommendations":recommendations,
        "final_state":state_to_labels(system,current_state),
        "final_score":calculate_state_score(system,current_state),
        "remaining_budget":round(float(available_budget),2),
        "ending_reason":ending_reason,
        "simulation_note":_simulation_note
    }

def get_system_features(system):
    return _system_features[system]