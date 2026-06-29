# src/logic/helpers.py
import pandas as pd
import numpy as np

def _add_polynomial_terms(dataframe, all_alphabet_vars):
    """
    Augments a dataframe with polynomial terms for response surface methodology.
    """
    # Add squared terms
    for header in all_alphabet_vars:
        squared_header = header + '_sq'
        dataframe[header] = pd.to_numeric(dataframe[header], errors='coerce')
        dataframe[squared_header] = dataframe[header] ** 2

    # Add interaction terms if there is more than one independent variable
    if len(all_alphabet_vars) > 1:
        for i in range(len(all_alphabet_vars)):
            for j in range(i + 1, len(all_alphabet_vars)):
                interaction_header = all_alphabet_vars[i] + '*' + all_alphabet_vars[j]
                dataframe[interaction_header] = dataframe[all_alphabet_vars[i]] * dataframe[all_alphabet_vars[j]]

def generate_prediction_grid(model, x_var, y_var, fixed_vars, x_range, y_range, n=100):
    """
    Creates a meshgrid for two variables and builds a flat DataFrame for model prediction.
    Returns (x_grid, y_grid, flat_predict_df) — not the predictions, so callers can customize what to predict.
    """
    x_values = np.linspace(x_range[0], x_range[1], n)
    y_values = np.linspace(y_range[0], y_range[1], n)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    flat_predict_df = pd.DataFrame({x_var: x_grid.flatten(), y_var: y_grid.flatten()})
    for var, val in fixed_vars.items():
        flat_predict_df[var] = val
    return x_grid, y_grid, flat_predict_df