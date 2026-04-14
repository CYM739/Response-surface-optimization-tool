# src/logic/models.py
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from sklearn.preprocessing import OrdinalEncoder
from scipy.optimize import minimize, basinhopping, shgo
from docx import Document
from docx.shared import Inches
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Categorical, Real
import io
from datetime import datetime
from itertools import combinations
import concurrent.futures
import warnings
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import io
from zipfile import ZipFile
from .helpers import _add_polynomial_terms

class OLSWrapper:
    def __init__(self, model, formula, independent_vars=None):
        if model is None:
            raise ValueError("Model cannot be None.")
        self.model = model
        self.formula = formula
        self.params = model.params
        if independent_vars is not None:
            self.independent_vars = list(independent_vars)
        else:
            self.independent_vars = [term for term in model.model.exog_names if '_' not in term and '*' not in term and ':' not in term and 'Intercept' not in term]
    def predict(self, dataframe):
        """
        Ensures polynomial terms are present before making a prediction.
        """
        predict_df = dataframe.copy()
        _add_polynomial_terms(predict_df, self.independent_vars)
        return self.model.predict(predict_df)

    def get_summary(self):
        return self.model.summary().as_text()

    def get_params_df(self):
        """Returns the model parameters as a DataFrame."""
        params_df = self.model.params.reset_index()
        params_df.columns = ['Term', 'Coefficient']
        return params_df

class SVRWrapper:
    def __init__(self, independent_vars):
        self.model = None
        self.independent_vars = independent_vars
        self.X_train = None
        self.y_train = None
        self.formula = "Non-linear SVR Model"
        self.params = None

    def fit(self, dataframe, dependent_var, C=1.0, gamma='scale'):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(C=C, gamma=gamma))
        ])
        self.X_train = dataframe[self.independent_vars]
        self.y_train = dataframe[dependent_var]
        self.pipeline.fit(self.X_train, self.y_train)
        self.params = self.pipeline.named_steps['svr'].get_params()

    def predict(self, dataframe):
        X_pred = dataframe[self.independent_vars]
        return self.pipeline.predict(X_pred)

    def get_summary(self):
        if self.X_train is not None and self.y_train is not None:
            r2_score = self.pipeline.score(self.X_train, self.y_train)
        else:
            r2_score = 'N/A'

        params = self.pipeline.named_steps['svr'].get_params()
        summary = (
            f"Support Vector Regression (SVR) Summary\n"
            f"-----------------------------------------\n"
            f"Kernel: {params['kernel']}\n"
            f"C (Regularization): {params['C']}\n"
            f"Gamma: {params['gamma']}\n"
            f"R-squared (on training data): {r2_score if isinstance(r2_score, str) else f'{r2_score:.4f}'}\n"
        )
        return summary

class RandomForestWrapper:
    def __init__(self, independent_vars):
        self.model = None
        self.independent_vars = independent_vars
        self.X_train = None
        self.y_train = None
        self.formula = "Non-linear Random Forest Model"
        self.params = None
        self.variable_descriptions = None

    def fit(self, dataframe, dependent_var, n_estimators=100, max_depth=10, random_state=42, variable_descriptions=None):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.X_train = dataframe[self.independent_vars]
        self.y_train = dataframe[dependent_var]
        self.model.fit(self.X_train, self.y_train)
        self.params = self.model.get_params()
        self.variable_descriptions = variable_descriptions

    def predict(self, dataframe):
        X_pred = dataframe[self.independent_vars]
        return self.model.predict(X_pred)

    def get_summary(self):
        if self.X_train is not None and self.y_train is not None:
            r2_score = self.model.score(self.X_train, self.y_train)
        else:
            r2_score = 'N/A'

        params = self.model.get_params()
        summary = (
            f"Random Forest Regressor Summary\n"
            f"----------------------------------\n"
            f"Number of Estimators: {params['n_estimators']}\n"
            f"Max Depth: {params['max_depth']}\n"
            f"R-squared (on training data): {r2_score if isinstance(r2_score, str) else f'{r2_score:.4f}'}\n"
        )

        if self.model and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if self.variable_descriptions:
                feature_names = [self.variable_descriptions.get(var, var) for var in self.independent_vars]
            else:
                feature_names = self.independent_vars

            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            summary += "\nFeature Importances:\n"
            summary += "--------------------\n"
            summary += feature_importance_df.to_string(index=False)

        return summary