# src/logic/models.py
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from sklearn.preprocessing import OrdinalEncoder
from docx import Document
from docx.shared import Inches
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime
import warnings
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit
from zipfile import ZipFile
from .helpers import _add_polynomial_terms

class OLSWrapper:
    def __init__(self, model, formula):
        if model is None:
            raise ValueError("Model cannot be None.")
        self.model = model
        self.formula = formula
        self.params = model.params
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


def _build_poly_term_names(independent_vars):
    """Term names in the order NonlinearLSWrapper stores coefficients:
    intercept, linear, squared, then pairwise interactions."""
    names = ['Intercept']
    names.extend(independent_vars)
    names.extend(f'{v}_sq' for v in independent_vars)
    for i, v1 in enumerate(independent_vars):
        for v2 in independent_vars[i + 1:]:
            names.append(f'{v1}*{v2}')
    return names


def _poly_eval(X, params):
    """Evaluate the quadratic response-surface polynomial used by both
    OLSWrapper and NonlinearLSWrapper:
        y = a0 + sum_i ai*xi + sum_i bi*xi^2 + sum_{i<j} cij*xi*xj
    X: (n_samples, n_features) ndarray. params: 1-D, length
    1 + 2n + n(n-1)/2 in the order produced by _build_poly_term_names.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n_samples, n_features = X.shape
    idx = 0
    y = np.full(n_samples, params[idx], dtype=float)
    idx += 1
    for i in range(n_features):
        y = y + params[idx] * X[:, i]
        idx += 1
    for i in range(n_features):
        y = y + params[idx] * X[:, i] ** 2
        idx += 1
    for i in range(n_features):
        for j in range(i + 1, n_features):
            y = y + params[idx] * X[:, i] * X[:, j]
            idx += 1
    return y


class NonlinearLSWrapper:
    """Matlab `fitnlm`-equivalent least-squares fit of the same quadratic
    response-surface polynomial that OLSWrapper uses.

    For the PRS polynomial the model is linear in parameters, so unconstrained
    NLS and OLS share the same convex objective (sum of squared residuals)
    and the same unique global minimum. Coefficients therefore match OLS to
    numerical tolerance (~1e-8) when `bounds` is None.

    The value this wrapper adds over OLSWrapper:
      - Parameter bounds (matches Matlab fitnlm's Lower/Upper options), which
        let the user constrain coefficient signs and prevent the quadratic
        surface from extrapolating to unphysical values.
      - Extension point for non-polynomial response surfaces in the future.

    Backend: scipy.optimize.curve_fit (Levenberg-Marquardt when unbounded,
    Trust-Region-Reflective when bounded). Same two algorithms Matlab `fitnlm`
    switches between.
    """

    def __init__(self, independent_vars):
        self.independent_vars = list(independent_vars)
        self.formula = None
        self.params = None
        self.param_names = _build_poly_term_names(self.independent_vars)
        self.param_cov = None
        self.r2_score = None
        self._bounds_used = None
        self._dependent_var = None
        self._n_obs = None
        self._scaler = None  # optional StandardScaler, populated when fit(scale_inputs=True)
        self._log_transform = False  # optional log-dose fit, populated when fit(log_transform=True)

    def fit(self, dataframe, dependent_var, bounds=None, p0=None, maxfev=10000,
            scale_inputs=False, log_transform=False):
        """Fit the polynomial to `dataframe[independent_vars]` vs
        `dataframe[dependent_var]`.

        bounds: None for unconstrained (reproduces OLS exactly, matches
          Matlab fitnlm default). Otherwise a (lower, upper) tuple of
          length-n_params arrays; use ±np.inf for one-sided bounds. When
          bounds are supplied, curve_fit switches from LM to
          Trust-Region-Reflective — matching Matlab fitnlm's behavior.
        p0: optional initial guess. Defaults to zeros — fine for
          linear-in-params problems.
        scale_inputs: standardize each independent variable (zero mean,
          unit variance) before forming polynomial terms. Recommended when
          the raw inputs span orders of magnitude (e.g., the IOR1 paper
          data: Adriamycin ~10^2 ng/mL vs Herceptin ~10^6), where unscaled
          x^2 and x*y terms produce a design matrix with condition number
          > 10^30. Predictions via `.predict(df)` automatically apply the
          same scaler, so callers interact with raw units.
        log_transform: fit the polynomial in log-dose space — i.e. apply
          np.log to each independent variable before (optional) scaling
          and polynomial expansion. Common in pharmacology where response
          is approximately linear in log-dose (Hill equation roots). All
          inputs must be strictly positive; ValueError is raised otherwise.
          When combined with scale_inputs=True, the pipeline is
          log -> standardize -> polynomial.

        Underdetermined data (n_obs < n_params): LM refuses to run — same
        restriction Matlab fitnlm imposes, and the reason the paper
        interpolated its 13 daily rows to 15 before fitting. We fall back
        to Trust-Region-Reflective with ±inf bounds, which does not have
        this restriction; the user gets a warning so they know the result
        is a minimum on an under-identified manifold.
        """
        X_raw = dataframe[self.independent_vars].values.astype(float)
        y = dataframe[dependent_var].values.astype(float)

        self._log_transform = bool(log_transform)
        if log_transform:
            if np.any(X_raw <= 0):
                raise ValueError(
                    "log_transform=True requires strictly positive inputs. "
                    "Offending columns: "
                    + ", ".join(
                        c for c, v in zip(self.independent_vars, X_raw.min(axis=0))
                        if v <= 0
                    )
                )
            X_transformed = np.log(X_raw)
        else:
            X_transformed = X_raw

        if scale_inputs:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X_transformed)
        else:
            self._scaler = None
            X = X_transformed

        n_params = len(self.param_names)
        n_obs = len(y)

        if p0 is None:
            p0 = np.zeros(n_params)

        def model_fn(X_arg, *params):
            return _poly_eval(X_arg, params)

        kwargs = dict(p0=p0, maxfev=maxfev)
        if bounds is not None:
            kwargs['bounds'] = bounds
            kwargs['method'] = 'trf'
        elif n_obs < n_params:
            warnings.warn(
                f"NonlinearLSWrapper: n_obs={n_obs} < n_params={n_params}. "
                f"Levenberg-Marquardt requires n_obs >= n_params (same as "
                f"Matlab fitnlm). Falling back to Trust-Region-Reflective "
                f"with no bounds — fit lives on an under-identified manifold.",
                RuntimeWarning,
            )
            kwargs['bounds'] = (
                np.full(n_params, -np.inf),
                np.full(n_params, np.inf),
            )
            kwargs['method'] = 'trf'

        popt, pcov = curve_fit(model_fn, X, y, **kwargs)

        self.params = popt
        self.param_cov = pcov
        self._bounds_used = bounds
        self._dependent_var = dependent_var
        self._n_obs = len(y)

        y_pred = _poly_eval(X, popt)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        self.r2_score = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        terms = [f"{c:+.4g}*{n}" for c, n in zip(popt, self.param_names)]
        self.formula = f"{dependent_var} ~ " + " ".join(terms)

    def predict(self, dataframe):
        if self.params is None:
            raise RuntimeError("NonlinearLSWrapper has not been fit yet.")
        X = dataframe[self.independent_vars].values.astype(float)
        if self._log_transform:
            if np.any(X <= 0):
                raise ValueError(
                    "predict() received non-positive values but model was fit "
                    "with log_transform=True."
                )
            X = np.log(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return _poly_eval(X, self.params)

    def get_params_df(self):
        stderr = np.sqrt(np.diag(self.param_cov)) if self.param_cov is not None else np.full_like(self.params, np.nan)
        return pd.DataFrame({
            'Term': self.param_names,
            'Coefficient': self.params,
            'Std Error': stderr,
        })

    def get_summary(self):
        lines = [
            "Nonlinear Least-Squares (fitnlm-equivalent) Summary",
            "---------------------------------------------------",
            f"Dependent variable : {self._dependent_var}",
            f"Backend            : scipy.optimize.curve_fit "
            f"({'Trust-Region-Reflective (bounded)' if self._bounds_used is not None else 'Levenberg-Marquardt'})",
            f"Observations       : {self._n_obs}",
            f"Input transform    : "
            f"{'log(x) + StandardScaler' if (self._log_transform and self._scaler is not None) else 'log(x)' if self._log_transform else 'StandardScaler' if self._scaler is not None else 'none (raw units)'}",
            f"Parameters         : {len(self.param_names)}",
            f"R-squared          : {self.r2_score:.4f}" if self.r2_score is not None else "R-squared          : N/A",
            "",
            "Coefficients",
            "------------",
        ]
        df = self.get_params_df()
        lines.append(df.to_string(index=False, float_format=lambda v: f"{v: .6g}"))
        return "\n".join(lines)


class RidgeWrapper:
    """Ridge (L2-regularized) regression over the same quadratic PRS
    polynomial that OLSWrapper and NonlinearLSWrapper use.

    Why this exists: on rank-deficient data (e.g., the IOR1 paper where 13
    rows of measurements have plateau regions that leave the design matrix
    effectively rank ~10 for a 15-coef polynomial), OLS picks an arbitrary
    point on a flat SSR manifold and fitnlm's LM refuses to run. Ridge
    penalizes the coefficient L2 norm, which pins down a unique minimum
    even under rank deficiency — and suppresses extrapolation pathology
    at the edges of the dose box.

    Defaults:
      - scale_inputs=True (Ridge is scale-sensitive; always scale unless
        you have a reason not to).
      - alpha=1e-3 (mild regularization; start here and raise if the
        response surface still looks extrapolatory).

    Interface matches OLS/SVR/RF wrappers (predict, get_summary,
    independent_vars, formula, params, r2_score).
    """

    def __init__(self, independent_vars):
        self.independent_vars = list(independent_vars)
        self.formula = None
        self.params = None
        self.param_names = _build_poly_term_names(self.independent_vars)
        self.model = None
        self.alpha = None
        self.r2_score = None
        self._scaler = None
        self._log_transform = False
        self._dependent_var = None
        self._n_obs = None

    def _build_feature_matrix(self, X_raw):
        """Produce the same polynomial feature columns used by
        NonlinearLSWrapper's _poly_eval, in the same order (linear,
        squared, pairwise interactions — intercept handled by sklearn)."""
        X_raw = np.asarray(X_raw, dtype=float)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)
        n_samples, n_features = X_raw.shape
        cols = [X_raw]
        cols.append(X_raw ** 2)
        interactions = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions.append((X_raw[:, i] * X_raw[:, j]).reshape(-1, 1))
        if interactions:
            cols.append(np.hstack(interactions))
        return np.hstack(cols)

    def fit(self, dataframe, dependent_var, alpha=1e-3, scale_inputs=True, log_transform=False):
        """Fit Ridge over polynomial features of the independent variables.

        alpha: L2 regularization strength. 0 recovers OLS (but use
          OLSWrapper / NonlinearLSWrapper for that — Ridge(alpha=0) misses
          the intercept conventions that statsmodels handles). Typical
          range: 1e-6 to 1e+2 depending on feature scale.
        scale_inputs: standardize independent vars before computing
          polynomial terms. Defaults True because Ridge is scale-sensitive.
        log_transform: fit in log-dose space (response is a polynomial
          function of log(drug concentration)). Common in pharmacology.
          All inputs must be strictly positive. Combined with
          scale_inputs=True, pipeline is: log -> standardize -> polynomial.
        """
        X_raw = dataframe[self.independent_vars].values.astype(float)
        y = dataframe[dependent_var].values.astype(float)

        self._log_transform = bool(log_transform)
        if log_transform:
            if np.any(X_raw <= 0):
                raise ValueError(
                    "log_transform=True requires strictly positive inputs. "
                    "Offending columns: "
                    + ", ".join(
                        c for c, v in zip(self.independent_vars, X_raw.min(axis=0))
                        if v <= 0
                    )
                )
            X_transformed = np.log(X_raw)
        else:
            X_transformed = X_raw

        if scale_inputs:
            self._scaler = StandardScaler()
            X_raw_s = self._scaler.fit_transform(X_transformed)
        else:
            self._scaler = None
            X_raw_s = X_transformed

        feat = self._build_feature_matrix(X_raw_s)

        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.model.fit(feat, y)

        # Store params in the same layout as NonlinearLSWrapper:
        # [intercept, linear..., squared..., interactions...]
        self.params = np.concatenate([[self.model.intercept_], self.model.coef_])
        self.alpha = alpha
        self._dependent_var = dependent_var
        self._n_obs = len(y)

        y_pred = self.model.predict(feat)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        self.r2_score = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        terms = [f"{c:+.4g}*{n}" for c, n in zip(self.params, self.param_names)]
        self.formula = f"{dependent_var} ~ " + " ".join(terms) + f"  [Ridge alpha={alpha}]"

    def predict(self, dataframe):
        if self.model is None:
            raise RuntimeError("RidgeWrapper has not been fit yet.")
        X = dataframe[self.independent_vars].values.astype(float)
        if self._log_transform:
            if np.any(X <= 0):
                raise ValueError(
                    "predict() received non-positive values but model was fit "
                    "with log_transform=True."
                )
            X = np.log(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        feat = self._build_feature_matrix(X)
        return self.model.predict(feat)

    def get_params_df(self):
        return pd.DataFrame({
            'Term': self.param_names,
            'Coefficient': self.params,
        })

    def get_summary(self):
        lines = [
            "Ridge Regression (L2-regularized polynomial) Summary",
            "-----------------------------------------------------",
            f"Dependent variable : {self._dependent_var}",
            f"Regularization     : alpha = {self.alpha}",
            f"Input transform    : "
            f"{'log(x) + StandardScaler' if (self._log_transform and self._scaler is not None) else 'log(x)' if self._log_transform else 'StandardScaler' if self._scaler is not None else 'none (raw units)'}",
            f"Observations       : {self._n_obs}",
            f"Parameters         : {len(self.param_names)}",
            f"R-squared          : {self.r2_score:.4f}" if self.r2_score is not None else "R-squared          : N/A",
            "",
            "Coefficients",
            "------------",
        ]
        df = self.get_params_df()
        lines.append(df.to_string(index=False, float_format=lambda v: f"{v: .6g}"))
        return "\n".join(lines)


class SVRWrapper:
    def __init__(self, independent_vars):
        self.model = None
        self.independent_vars = independent_vars
        self.formula = "Non-linear SVR Model"
        self.params = None

    def fit(self, dataframe, dependent_var, C=1.0, gamma='scale'):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(C=C, gamma=gamma))
        ])
        X = dataframe[self.independent_vars]
        y = dataframe[dependent_var]
        self.pipeline.fit(X, y)
        self.r2_score = self.pipeline.score(X, y)
        self.params = self.pipeline.named_steps['svr'].get_params()

    def predict(self, dataframe):
        X_pred = dataframe[self.independent_vars]
        return self.pipeline.predict(X_pred)

    def get_summary(self):
        r2_score = getattr(self, 'r2_score', 'N/A')

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
        self.formula = "Non-linear Random Forest Model"
        self.params = None
        self.variable_descriptions = None

    def fit(self, dataframe, dependent_var, n_estimators=100, max_depth=10, random_state=42, variable_descriptions=None):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        X = dataframe[self.independent_vars]
        y = dataframe[dependent_var]
        self.model.fit(X, y)
        self.r2_score = self.model.score(X, y)
        self.params = self.model.get_params()
        self.variable_descriptions = variable_descriptions

    def predict(self, dataframe):
        X_pred = dataframe[self.independent_vars]
        return self.model.predict(X_pred)

    def get_summary(self):
        r2_score = getattr(self, 'r2_score', 'N/A')

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


class BRAIDWrapper:
    """Wraps BRAID surface parameters into the same interface as OLS/SVR/RF wrappers."""

    def __init__(self, params, drug1_name, drug2_name):
        self.params = params  # dict with EC50_1, h1, EC50_2, h2, E0, Einf, kappa, ...
        self.independent_vars = [drug1_name, drug2_name]
        self.formula = f"BRAID({drug1_name}, {drug2_name})"
        self._drug1 = drug1_name
        self._drug2 = drug2_name

    def predict(self, dataframe):
        D1 = dataframe[self._drug1].values.astype(float)
        D2 = dataframe[self._drug2].values.astype(float)
        EC1, EC2 = self.params["EC50_1"], self.params["EC50_2"]
        h1, h2 = self.params["h1"], self.params["h2"]
        E0, Einf = self.params["E0"], self.params["Einf"]
        kappa = self.params["kappa"]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            T1 = np.where(D1 > 0, np.clip((D1 / EC1) ** h1, 0, 1e6), 0.0)
            T2 = np.where(D2 > 0, np.clip((D2 / EC2) ** h2, 0, 1e6), 0.0)
            denom = np.clip(1.0 + T1 + T2 + kappa * T1 * T2, 1e-9, None)
            V = Einf + (E0 - Einf) / denom
        return V

    def get_summary(self):
        p = self.params
        kappa = p["kappa"]
        if kappa > 1:
            label = "Synergistic"
        elif kappa < -1:
            label = "Antagonistic"
        else:
            label = "Additive"
        return (
            f"BRAID Surface Model\n"
            f"-------------------\n"
            f"Drugs: {self._drug1} + {self._drug2}\n"
            f"Cell Line: {p.get('cell_line', 'N/A')}\n"
            f"EC50_1: {p['EC50_1']:.4g}, h1: {p['h1']:.3f}\n"
            f"EC50_2: {p['EC50_2']:.4g}, h2: {p['h2']:.3f}\n"
            f"E0: {p['E0']:.1f}%, Einf: {p['Einf']:.1f}%\n"
            f"kappa: {kappa:.3f} ({label})\n"
            f"R²: {p.get('r2_braid', 'N/A')}\n"
        )


class MechanisticWrapper:
    """Mechanistic 2-drug response-surface model (MuSyC or BRAID), fitted to the
    user's combination data via the `synergy` package.

    Unlike the polynomial wrappers, the surface is bounded by construction (it
    asymptotes between E0 and the combination's max effect), so it cannot
    extrapolate to impossible values (e.g. viability < 0). Synergy is read
    directly from the fitted parameters — MuSyC's alpha (potency) / gamma
    (cooperativity), or BRAID's kappa — rather than from a difference-to-null.

    Two drugs only. Implements the standard wrapper interface
    (predict, get_summary, get_params_df, independent_vars, formula, r2_score).
    """

    def __init__(self, independent_vars, kind="musyc"):
        ivs = list(independent_vars)
        if len(ivs) != 2:
            raise ValueError(
                "Mechanistic surface models (MuSyC / BRAID) support exactly two "
                f"drugs; got {len(ivs)}. Use OLS / Ridge / fitnlm for 1 or 3+ drugs."
            )
        self.independent_vars = ivs
        self.kind = kind.lower()
        self._drug1, self._drug2 = ivs
        self.model = None
        self.params = {}
        self.r2_score = None
        self._dependent_var = None
        self.formula = f"{self.kind.upper()}({self._drug1}, {self._drug2})"

    def fit(self, dataframe, dependent_var, **kwargs):
        from synergy.combination import MuSyC, BRAID
        d1 = dataframe[self._drug1].to_numpy(dtype=float)
        d2 = dataframe[self._drug2].to_numpy(dtype=float)
        E = dataframe[dependent_var].to_numpy(dtype=float)
        self.model = MuSyC() if self.kind == "musyc" else BRAID()
        self.model.fit(d1, d2, E)
        self._dependent_var = dependent_var
        try:
            p = self.model.get_parameters()
            self.params = dict(p) if hasattr(p, "items") else {f"p{i}": v for i, v in enumerate(p)}
        except Exception:
            self.params = {}
        pred = np.asarray(self.model.E(d1, d2), dtype=float)
        ss_res = float(np.sum((E - pred) ** 2))
        ss_tot = float(np.sum((E - E.mean()) ** 2))
        self.r2_score = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def predict(self, dataframe):
        if self.model is None:
            raise RuntimeError("MechanisticWrapper has not been fit yet.")
        d1 = dataframe[self._drug1].to_numpy(dtype=float)
        d2 = dataframe[self._drug2].to_numpy(dtype=float)
        return np.asarray(self.model.E(d1, d2), dtype=float)

    def get_params_df(self):
        return pd.DataFrame({"Term": list(self.params.keys()),
                             "Coefficient": [float(v) for v in self.params.values()]})

    def _synergy_label(self):
        p = self.params
        if self.kind == "braid":
            k = float(p.get("kappa", 0.0))
            tag = "Synergistic" if k > 0.1 else "Antagonistic" if k < -0.1 else "Additive"
            return f"kappa = {k:.3f} ({tag})"
        a12, a21 = float(p.get("alpha12", 1.0)), float(p.get("alpha21", 1.0))
        amean = 0.5 * (a12 + a21)
        tag = "Synergistic" if amean > 1.1 else "Antagonistic" if amean < 0.9 else "Additive"
        return f"alpha12 = {a12:.2f}, alpha21 = {a21:.2f} (potency synergy: {tag})"

    def get_summary(self):
        lines = [
            f"{self.kind.upper()} mechanistic surface ({self._drug1} + {self._drug2})",
            "-" * 48,
            f"Dependent variable : {self._dependent_var}",
            (f"R-squared          : {self.r2_score:.4f}"
             if self.r2_score is not None else "R-squared          : N/A"),
            f"Synergy            : {self._synergy_label()}",
            "",
            "Parameters",
            "----------",
            self.get_params_df().to_string(index=False, float_format=lambda v: f"{v: .4g}"),
        ]
        return "\n".join(lines)


# ── Model capability flags ──────────────────────────────────────────────────────
# Views gate themselves on these instead of isinstance() checks, so adding a new
# model type only requires declaring its capabilities here.
#   IS_PARAMETRIC_POLYNOMIAL : quadratic-polynomial regression -> Diagnostics tab
#                              (residual-based assumption tests apply)
#   SUPPORTS_GRADIENT_OPT    : smooth surface -> standard (SciPy) Optimizer tab
#                              (non-smooth models must use the AI/Bayesian optimizer)
#   SUPPORTS_OLS_INFERENCE   : full statsmodels inference (p-values, bootstrap CIs,
#                              downloadable reports, Drug Elimination scoring)
OLSWrapper.IS_PARAMETRIC_POLYNOMIAL = True
OLSWrapper.SUPPORTS_GRADIENT_OPT = True
OLSWrapper.SUPPORTS_OLS_INFERENCE = True

NonlinearLSWrapper.IS_PARAMETRIC_POLYNOMIAL = True
NonlinearLSWrapper.SUPPORTS_GRADIENT_OPT = True
NonlinearLSWrapper.SUPPORTS_OLS_INFERENCE = False

RidgeWrapper.IS_PARAMETRIC_POLYNOMIAL = True
RidgeWrapper.SUPPORTS_GRADIENT_OPT = True
RidgeWrapper.SUPPORTS_OLS_INFERENCE = False

SVRWrapper.IS_PARAMETRIC_POLYNOMIAL = False
SVRWrapper.SUPPORTS_GRADIENT_OPT = True
SVRWrapper.SUPPORTS_OLS_INFERENCE = False

RandomForestWrapper.IS_PARAMETRIC_POLYNOMIAL = False
RandomForestWrapper.SUPPORTS_GRADIENT_OPT = False
RandomForestWrapper.SUPPORTS_OLS_INFERENCE = False

BRAIDWrapper.IS_PARAMETRIC_POLYNOMIAL = False
BRAIDWrapper.SUPPORTS_GRADIENT_OPT = True
BRAIDWrapper.SUPPORTS_OLS_INFERENCE = False

MechanisticWrapper.IS_PARAMETRIC_POLYNOMIAL = False   # not a polynomial -> no OLS diagnostics
MechanisticWrapper.SUPPORTS_GRADIENT_OPT = True       # smooth, bounded surface
MechanisticWrapper.SUPPORTS_OLS_INFERENCE = False
