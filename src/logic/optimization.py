# src/logic/optimization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping, shgo
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Categorical, Real
from itertools import combinations
import concurrent.futures
import warnings
from .models import OLSWrapper
from .data_processing import _add_polynomial_terms

def objective_function(x, model, all_alphabet_vars):
    predict_df = pd.DataFrame([x], columns=all_alphabet_vars)
    prediction = model.predict(predict_df)
    return prediction[0] if isinstance(prediction, (np.ndarray, pd.Series)) else prediction

def run_optimization(fun, bounds, start_points, constraints, algorithm, algo_params, args_tuple=()):
    """
    A centralized wrapper to execute different optimization algorithms from SciPy.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step, clipping to bounds",
            category=RuntimeWarning
        )
        if algorithm.startswith('SLSQP'):
            return minimize(fun=fun, x0=start_points, method='SLSQP', bounds=bounds, constraints=constraints, args=args_tuple)
        elif algorithm.startswith('Basinhopping'):
            minimizer_kwargs = {'method': 'SLSQP', 'bounds': bounds, 'constraints': constraints, 'args': args_tuple}
            return basinhopping(func=fun, x0=start_points, niter=algo_params['niter'], minimizer_kwargs=minimizer_kwargs)
        elif algorithm.startswith('SHGO'):
            return shgo(func=fun, bounds=bounds, constraints=constraints, n=algo_params.get('shgo_n', 100), iters=algo_params.get('shgo_iters', 3), args=args_tuple)
def create_range_constraints(model, independent_vars, r_min, r_max):
    """
    Creates a list of inequality constraints for scipy.optimize.
    Constraint format: fun(x) >= 0.
    """
    return [
        {'type': 'ineq', 'fun': lambda x: objective_function(x, model, independent_vars) - r_min},
        {'type': 'ineq', 'fun': lambda x: r_max - objective_function(x, model, independent_vars)}
    ]
    
def run_weighted_tradeoff_optimization(model_1, model_2, independent_vars, bounds, start_points,
                                       r_min_1, r_max_1, r_min_2, r_max_2,
                                       weights, algorithm, algo_params):
    """
    Stage 3: Finds the best trade-off by Maximizing a weighted score 
    subject to STRICT constraints on both models.
    """
    # 1. Create Strict Constraints for both models
    constraints = []
    constraints.extend(create_range_constraints(model_1, independent_vars, r_min_1, r_max_1))
    constraints.extend(create_range_constraints(model_2, independent_vars, r_min_2, r_max_2))

    # 2. Define Objective (Maximize Weighted Score)
    # Scipy minimizes, so we return negative weighted score.
    w1 = weights.get('model_1', 0.5)
    w2 = weights.get('model_2', 0.5)

    def weighted_objective(x):
        val_1 = objective_function(x, model_1, independent_vars)
        val_2 = objective_function(x, model_2, independent_vars)
        # Score = w1*Model1 + w2*Model2
        score = (val_1 * w1) + (val_2 * w2)
        return -score

    # 3. Run Optimization
    result = run_optimization(weighted_objective, bounds, start_points, constraints, algorithm, algo_params)
    return result
                                           
def run_multi_objective_penalty_optimization(model_1, model_2, independent_vars, bounds, start_points, 
                                             r_min_1, r_max_1, r_min_2, r_max_2, 
                                             algorithm, algo_params, penalty_weight=1000):
    """
    Stage 1: Multi-objective optimization using a penalty function.
    Finds a feasible starting point where BOTH models are within their specified ranges.
    """
    def objective_function_penalty(x):
        outcome_1 = objective_function(x, model_1, independent_vars)
        outcome_2 = objective_function(x, model_2, independent_vars)

        penalty = 0
        
        # Penalize Model 1 deviations
        if outcome_1 < r_min_1: penalty += (r_min_1 - outcome_1)**2
        elif outcome_1 > r_max_1: penalty += (outcome_1 - r_max_1)**2

        # Penalize Model 2 deviations
        if outcome_2 < r_min_2: penalty += (r_min_2 - outcome_2)**2
        elif outcome_2 > r_max_2: penalty += (outcome_2 - r_max_2)**2

        # We minimize pure penalty to find a valid entry point into the feasible region
        return penalty * penalty_weight

    result = run_optimization(objective_function_penalty, bounds, start_points, [], algorithm, algo_params)
    return result
    
def difference_objective_function(x, model_1, model_2, all_alphabet_vars, target_model_goal='Maximize'):
    """
    Calculates the difference between two model predictions, oriented by the optimization goal for model_2.
    The optimizer will then minimize the returned value to find the best trade-off.
    """
    prediction_1 = objective_function(x, model_1, all_alphabet_vars)
    prediction_2 = objective_function(x, model_2, all_alphabet_vars)

    if target_model_goal == 'Maximize':
        return prediction_1 - prediction_2
    else: 
        return prediction_2 - prediction_1
        
def _optimize_grid_point(args):
    """
    Worker function for parallel grid search optimization.
    """
    (target, model_1, model_2, independent_vars, bounds, start_points, algorithm, algo_params) = args
    
    constraint = {'type': 'eq', 'fun': lambda x: objective_function(x, model_1, independent_vars) - target}
    obj_fun = lambda x: -objective_function(x, model_2, independent_vars) # Maximize
    
    result = run_optimization(obj_fun, bounds, start_points, [constraint], algorithm, algo_params)
    
    return result if result.success else None

def run_grid_search_optimization(model_1, model_2, independent_vars, bounds, start_points, algorithm, algo_params, initial_solution, weights, tolerance, num_points=20):
    """
    Performs a grid search around the initial solution to find the top 5 weighted solutions.
    """
    score_components = [
        {"model_name": "model_1", "goal": "Maximize", "weight": weights['model_1'], "type": "model"},
        {"model_name": "model_2", "goal": "Maximize", "weight": weights['model_2'], "type": "model"},
        {"factor_name": "Total Dosage", "goal": "Minimize", "weight": weights['total_dosage'], "type": "factor"}
    ]
    wrapped_models = {"model_1": model_1, "model_2": model_2}

    def weighted_objective(x):
        return weighted_score_objective_function(x, score_components, wrapped_models, independent_vars)

    initial_outcome_1 = objective_function(initial_solution.x, model_1, independent_vars)
    lower_bound = initial_outcome_1 * (1 - tolerance)
    upper_bound = initial_outcome_1 * (1 + tolerance)
    target_outcomes = np.linspace(lower_bound, upper_bound, num_points)

    all_solutions = []
    
    tasks_args = [
        (target, model_1, model_2, independent_vars, bounds, start_points, algorithm, algo_params)
        for target in target_outcomes
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(_optimize_grid_point, tasks_args)
        all_solutions = [res for res in results if res is not None]

    for sol in all_solutions:
        sol.fun = weighted_objective(sol.x)

    sorted_results = sorted(all_solutions, key=lambda r: r.fun)

    top_5_solutions = []
    for res in sorted_results:
        is_duplicate = any(np.allclose(res.x, sol.x) for sol in top_5_solutions)
        if not is_duplicate:
            top_5_solutions.append(res)
        if len(top_5_solutions) == 5:
            break

    return top_5_solutions

def weighted_score_objective_function(x, score_components, wrapped_models, independent_vars):
    """
    Calculates a weighted score based on multiple model outcomes and other factors.
    """
    score = 0
    for comp in score_components:
        if comp['type'] == 'model':
            model = wrapped_models[comp['model_name']]
            outcome = objective_function(x, model, independent_vars)
            if comp['goal'] == 'Maximize':
                outcome = -outcome
            score += outcome * comp['weight']
        elif comp['type'] == 'factor':
            if comp['factor_name'] == 'Total Dosage':
                total_dosage = sum(x)
                score += total_dosage * comp['weight']
    return score

def run_bayesian_optimization(OLS_model, all_alphabet_vars, bounds, goal, n_calls,
                              n_initial_points, variable_descriptions, discrete_vars=None, target_value=None):
    """
    Performs Bayesian Optimization using scikit-optimize (`skopt`).
    """
    if discrete_vars is None:
        discrete_vars = {}

    dimensions = []
    for i, var_name in enumerate(all_alphabet_vars):
        if var_name in discrete_vars:
            allowed_values = [int(v) for v in discrete_vars[var_name]]
            dimensions.append(Categorical(categories=allowed_values, name=var_name))
        else:
            min_bound, max_bound = bounds[i]
            dimensions.append(Real(low=min_bound, high=max_bound, name=var_name))

    def bayesian_objective(x):
        """The objective function for the Bayesian optimizer to minimize."""
        prediction = objective_function(x, OLS_model, all_alphabet_vars)
        if target_value is not None:
            return (prediction - target_value) ** 2
        elif goal == "Minimize":
            return prediction**2 + 1e6 if prediction < 0 else prediction
        else: # Maximize
            return -prediction

    result = gp_minimize(
        func=bayesian_objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=0
    )

    best_dosages = result.x
    best_outcome = objective_function(best_dosages, OLS_model, all_alphabet_vars)

    convergence_fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_convergence(result, ax=ax)
    ax.set_title("Convergence Plot")
    ax.set_ylabel("Best Objective Value Found")
    convergence_fig.tight_layout()

    descriptive_names = [variable_descriptions.get(v, v) for v in all_alphabet_vars]
    plot_obj = plot_objective(result, dimensions=descriptive_names)
    objective_fig = plot_obj.figure if hasattr(plot_obj, 'figure') else plot_obj
    objective_fig.tight_layout()

    return {
        "dosages": best_dosages,
        "outcome": best_outcome,
        "convergence_plot": convergence_fig,
        "objective_plot": objective_fig,
        "raw_result": result
    }

def run_bayesian_optimization_constrained(
    objective_model,
    constraint_model,
    all_alphabet_vars,
    bounds,
    objective_goal,
    constraint_target,
    n_calls,
    n_initial_points,
    variable_descriptions,
    fallback_goal,
    constraint_tolerance=1e-6
):
    """
    Performs constrained Bayesian Optimization using a penalty method.
    """

    def constrained_objective(x):
        """The objective function with an integrated penalty for constraint violations."""
        constraint_prediction = objective_function(x, constraint_model, all_alphabet_vars)

        constraint_met = False
        if fallback_goal == "Value above target":
            if constraint_prediction >= constraint_target - constraint_tolerance:
                constraint_met = True
        else:  # "Value below target"
            if constraint_prediction <= constraint_target + constraint_tolerance:
                constraint_met = True

        if constraint_met:
            objective_prediction = objective_function(x, objective_model, all_alphabet_vars)
            return objective_prediction if objective_goal == "Minimize" else -objective_prediction
        else:
            return (constraint_prediction - constraint_target)**2 + 1e6

    result = gp_minimize(
        func=constrained_objective,
        dimensions=bounds,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=0
    )

    best_dosages = result.x
    final_objective_outcome = objective_function(best_dosages, objective_model, all_alphabet_vars)
    final_constraint_outcome = objective_function(best_dosages, constraint_model, all_alphabet_vars)

    final_constraint_met = False
    if fallback_goal == "Value above target":
        if final_constraint_outcome >= constraint_target - constraint_tolerance:
            final_constraint_met = True
    else: # "Value below target"
        if final_constraint_outcome <= constraint_target + constraint_tolerance:
            final_constraint_met = True

    if abs(final_constraint_outcome - constraint_target) < 1e-9:
        final_constraint_outcome = constraint_target

    convergence_fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_convergence(result, ax=ax)
    ax.set_title("Convergence Plot")
    ax.set_ylabel("Best Objective Value Found")
    convergence_fig.tight_layout()

    descriptive_names = [variable_descriptions.get(v, v) for v in all_alphabet_vars]
    plot_obj = plot_objective(result, dimensions=descriptive_names)
    objective_fig = plot_obj.figure if hasattr(plot_obj, 'figure') else plot_obj
    objective_fig.tight_layout()

    return {
        "success": final_constraint_met, "dosages": best_dosages,
        "objective_outcome": final_objective_outcome, "constraint_outcome": final_constraint_outcome,
        "convergence_plot": convergence_fig, "objective_plot": objective_fig, "raw_result": result
    }

def _optimize_single_combination(args):
    """
    A dedicated worker function for parallel combination analysis.
    """
    (combo, OLS_model, all_independent_vars, variable_stats,
     goal, target_value, n_calls, n_initial_points, variable_descriptions, outcome_min) = args

    fixed_vars = {
        var: (variable_stats[var][0] + variable_stats[var][2]) / 2
        for var in all_independent_vars if var not in combo
    }

    def combo_objective_function(x):
        full_input_dict = fixed_vars.copy()
        for var_name, var_value in zip(combo, x):
            full_input_dict[var_name] = var_value

        full_input_vector = [full_input_dict[var] for var in all_independent_vars]
        prediction = objective_function(full_input_vector, OLS_model, all_independent_vars)

        penalty = 0

        # Penalise outcomes below the user-defined minimum (mirrors r_min in the normal optimizer)
        if outcome_min is not None and prediction < outcome_min:
            penalty += 1e6 + (outcome_min - prediction) * 1e4

        objective_value = 0
        if target_value is not None:
            objective_value = (prediction - target_value) ** 2
        elif goal == "Minimize":
            objective_value = prediction
        else: # Maximize
            objective_value = -prediction

        return objective_value + penalty

    # If the minimum dose for a variable is 0, use the second-minimum (lowest non-zero
    # value actually tested) as the lower bound. This hard-excludes zero from the Bayesian
    # search space, which is more reliable than a penalty on a GP surrogate.
    combo_bounds = []
    for var in combo:
        lo, second_lo, hi = variable_stats[var]
        if lo == 0 and second_lo > 0:
            lo = second_lo
        combo_bounds.append((lo, hi))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = gp_minimize(
            func=combo_objective_function, dimensions=combo_bounds,
            n_calls=n_calls, n_initial_points=n_initial_points, random_state=0
        )

    best_dosages_for_combo = result.x
    final_input_dict = fixed_vars.copy()
    for var_name, var_value in zip(combo, best_dosages_for_combo):
        final_input_dict[var_name] = var_value
    final_input_vector = [final_input_dict[var] for var in all_independent_vars]
    final_outcome = objective_function(final_input_vector, OLS_model, all_independent_vars)

    descriptive_combo_names = [variable_descriptions.get(c, c) for c in combo]

    return {
        "combination": combo, "descriptive_combination": descriptive_combo_names,
        "outcome": final_outcome, "dosages": dict(zip(combo, best_dosages_for_combo)),
        "descriptive_dosages": dict(zip(descriptive_combo_names, best_dosages_for_combo))
    }

def run_combination_optimization(OLS_model, all_independent_vars, variable_stats, combo_size,
                                 goal, target_value, n_calls, n_initial_points,
                                 variable_descriptions, progress_bar, status_text,
                                 outcome_min=None):
    """
    Orchestrates the parallel Bayesian optimization of all variable combinations.
    """
    all_combos = list(combinations(all_independent_vars, combo_size))
    all_results = []

    tasks_args = [
        (combo, OLS_model, all_independent_vars, variable_stats,
         goal, target_value, n_calls, n_initial_points, variable_descriptions, outcome_min)
        for combo in all_combos
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_optimize_single_combination, args) for args in tasks_args]
        total_tasks = len(futures)
        completed_tasks = 0

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"A combination failed to optimize: {e}")

            completed_tasks += 1
            progress = completed_tasks / total_tasks
            status_text.text(f"Optimizing combinations... {completed_tasks}/{total_tasks} complete.")
            progress_bar.progress(progress)

    floor = outcome_min if outcome_min is not None else 0
    if goal == 'Minimize' or target_value is not None:
        valid_results = [r for r in all_results if r['outcome'] >= floor]
        if valid_results:
            key_func = lambda r: abs(r['outcome'] - target_value) if target_value is not None else r['outcome']
            sorted_results = sorted(valid_results, key=key_func)
            invalid_results = sorted([r for r in all_results if r['outcome'] < floor], key=lambda r: abs(r['outcome']))
            sorted_results.extend(invalid_results)
        else:
            sorted_results = sorted(all_results, key=lambda r: abs(r['outcome']))
    else: # Maximize
        valid_results = [r for r in all_results if r['outcome'] >= floor]
        invalid_results = [r for r in all_results if r['outcome'] < floor]
        sorted_results = (
            sorted(valid_results, key=lambda r: r['outcome'], reverse=True)
            + sorted(invalid_results, key=lambda r: r['outcome'], reverse=True)
        )

    best_combination_info = sorted_results[0] if sorted_results else None

    ranking_df = pd.DataFrame({
        'Combination': [' & '.join(r['descriptive_combination']) for r in all_results],
        'Outcome': [r['outcome'] for r in all_results]
    })
    return {
        "best_combination": best_combination_info,
        "sorted_results": sorted_results,
        "ranking_df": ranking_df
    }

def run_classic_multi_objective_optimization(
    model_1, model_2, independent_vars, bounds, start_points,
    r_min, r_max, target_model_goal, algorithm, algo_params
):
    """
    This is the restored multi-objective optimization logic from the original script.
    It uses a two-stage approach with an auto-relaxing constraint.
    """
    # =====================================================================================
    # STAGE 1: OPTIMIZE TARGET CELL OUTCOME (WITH ENHANCED AUTO-RELAX)
    # =====================================================================================
    if target_model_goal == 'Maximize':
        def objective_function_stage1(x):
            return -objective_function(x, model_2, independent_vars)
    else:  # 'Minimize'
        def objective_function_stage1(x):
            return objective_function(x, model_2, independent_vars)

    best_result_stage1 = None
    solution_found_stage1 = False
    max_attempts = 6
    current_R_min, current_R_max = r_min, r_max

    for attempt in range(max_attempts):
        if attempt > 0:
            adjustment = 0.1 * attempt
            current_R_min = max(0.0, r_min - adjustment)
            current_R_max = min(1.0, r_max + adjustment)

        def constraint_func(x):
            outcome = objective_function(x, model_1, independent_vars)
            return np.array([outcome - current_R_min, current_R_max - outcome])
        
        constraint_config = {'type': 'ineq', 'fun': constraint_func}

        res = run_optimization(
            objective_function_stage1, bounds, start_points, [constraint_config],
            algorithm, algo_params
        )

        if res.success:
            best_result_stage1 = res
            solution_found_stage1 = True
            if attempt > 0:
                print(f"Solution found after relaxing constraints to: [{current_R_min:.2f}, {current_R_max:.2f}]")
            break

    if not solution_found_stage1:
        return None, "Stage 1 Failed: Could not find a viable solution even after relaxing constraints."

    target_outcome_s1 = -best_result_stage1.fun if target_model_goal == 'Maximize' else best_result_stage1.fun

    # =====================================================================================
    # STAGE 2: FIND DOSE-EFFICIENT PRECISE SOLUTION
    # =====================================================================================

    def minimize_total_dosage(x):
        return sum(x)

    def cell_1_constraint(x):
        outcome = objective_function(x, model_1, independent_vars)
        return np.array([outcome - current_R_min, current_R_max - outcome])

    if target_model_goal == 'Maximize':
        target_efficacy_threshold = target_outcome_s1 * 0.95
        def efficacy_constraint(x):
            outcome = objective_function(x, model_2, independent_vars)
            return outcome - target_efficacy_threshold
    else:
        target_efficacy_threshold = target_outcome_s1 * 1.05 if target_outcome_s1 > 0.01 else target_outcome_s1 * 0.95
        def efficacy_constraint(x):
            outcome = objective_function(x, model_2, independent_vars)
            return target_efficacy_threshold - outcome

    constraints_stage2 = [
        {'type': 'ineq', 'fun': cell_1_constraint},
        {'type': 'ineq', 'fun': efficacy_constraint}
    ]

    dose_optimization_result = run_optimization(
        minimize_total_dosage, bounds, best_result_stage1.x, constraints_stage2,
        algorithm, algo_params
    )
    
    # If dose optimization fails, fall back to the result from stage 1
    if dose_optimization_result.success:
        return dose_optimization_result.x, "Success"
    else:
        return best_result_stage1.x, "Success (dose optimization failed, returning stage 1 result)"
