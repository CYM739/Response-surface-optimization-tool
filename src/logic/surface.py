# src/logic/surface.py
"""
Unified Surface abstraction for response surface analysis.

All downstream operations (plotting, optimization, synergy, comparison)
work through this interface regardless of whether the surface comes from:
  - OLS / SVR / Random Forest model fitted to user data
  - BRAID lookup (pre-fitted 216k surfaces)
  - BRAID neural network prediction (v19 model)
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize


class Surface(ABC):
    """
    Abstract response surface.

    Implementations must define predict() and the property accessors.
    Grid generation, optimization, and synergy are provided as defaults
    that call predict() — subclasses can override for efficiency.
    """

    # ── Required interface ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def var_names(self) -> list:
        """Ordered list of independent variable names (e.g. ['Drug_A', 'Drug_B'])."""

    @property
    @abstractmethod
    def response_name(self) -> str:
        """Name of the response variable (e.g. 'Viability (%)')."""

    @property
    @abstractmethod
    def bounds(self) -> dict:
        """Variable bounds: {var_name: (low, high)} for each var in var_names."""

    @property
    @abstractmethod
    def params_table(self) -> list:
        """List of (param_name, value_str) tuples for display."""

    @property
    @abstractmethod
    def source_label(self) -> str:
        """Short label describing the surface origin (e.g. 'OLS fit', 'BRAID lookup')."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict response for each row in df.

        df has columns matching var_names (at minimum).
        Returns 1-D numpy array of length len(df).
        """

    @abstractmethod
    def summary(self) -> str:
        """Human-readable summary string."""

    # ── Default implementations ──────────────────────────────────────────────

    def predict_point(self, **doses) -> float:
        """Predict at a single point. E.g. surface.predict_point(Drug_A=1.0, Drug_B=2.0)."""
        df = pd.DataFrame([doses])
        return float(self.predict(df)[0])

    def predict_grid(self, var1, var2, n_points=60, fixed=None):
        """
        Generate a 2-D prediction grid over var1 × var2.

        Args:
            var1, var2: variable names (must be in var_names)
            n_points:   grid resolution per axis
            fixed:      dict of {var: value} for other variables (defaults to midpoint of bounds)

        Returns:
            (arr1, arr2, Z_grid)
            arr1: 1-D array [n_points] for var1
            arr2: 1-D array [n_points] for var2
            Z_grid: 2-D array [n_points, n_points] of predictions
        """
        b1 = self.bounds[var1]
        b2 = self.bounds[var2]
        arr1 = np.linspace(b1[0], b1[1], n_points)
        arr2 = np.linspace(b2[0], b2[1], n_points)
        G1, G2 = np.meshgrid(arr1, arr2)

        # Build flat prediction DataFrame
        flat = pd.DataFrame({var1: G1.ravel(), var2: G2.ravel()})
        if fixed is None:
            fixed = {}
        for v in self.var_names:
            if v not in (var1, var2):
                mid = fixed.get(v, (self.bounds[v][0] + self.bounds[v][1]) / 2)
                flat[v] = mid

        Z_flat = self.predict(flat)
        Z_grid = Z_flat.reshape(n_points, n_points)
        return arr1, arr2, Z_grid

    def optimize(self, objective="minimize", target_value=None,
                 fixed=None, dose_weight=None):
        """
        Optimize the surface response.

        Args:
            objective: 'minimize', 'maximize', or 'target'
            target_value: target response value (required if objective='target')
            fixed: dict {var: value} to fix certain variables
            dose_weight: dict {var: weight} for min-dose style optimization

        Returns:
            dict with {var_name: optimal_value, ..., 'response': predicted_value,
                       'success': bool}
        """
        free_vars = [v for v in self.var_names if v not in (fixed or {})]
        free_bounds = [self.bounds[v] for v in free_vars]

        def _obj(x):
            doses = dict(zip(free_vars, x))
            if fixed:
                doses.update(fixed)
            df = pd.DataFrame([doses])
            pred = float(self.predict(df)[0])
            if objective == "minimize":
                return pred
            elif objective == "maximize":
                return -pred
            else:  # target
                return (pred - target_value) ** 2

        # Multi-start
        starts = []
        for _ in range(6):
            x0 = [np.random.uniform(lo, hi) for lo, hi in free_bounds]
            starts.append(x0)
        # Also try midpoints
        starts.append([(lo + hi) / 2 for lo, hi in free_bounds])

        best = None
        for x0 in starts:
            try:
                res = scipy_minimize(_obj, x0, method="L-BFGS-B", bounds=free_bounds,
                                     options={"maxiter": 2000})
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:
                continue

        if best is None:
            return {"success": False}

        result = {v: float(best.x[i]) for i, v in enumerate(free_vars)}
        if fixed:
            result.update(fixed)
        result["response"] = self.predict_point(**{v: result[v] for v in self.var_names})
        result["success"] = True
        return result

    def synergy_grid(self, var1, var2, n_points=50, fixed=None, method="hsa"):
        """
        Compute synergy scores on a 2-D grid (for two-drug surfaces).

        Methods: 'hsa' (Highest Single Agent), 'bliss' (Bliss Independence)

        Returns:
            (arr1, arr2, response_grid, synergy_grid)
        """
        arr1, arr2, Z_combo = self.predict_grid(var1, var2, n_points, fixed)

        # Single-agent predictions: fix one drug at 0
        fixed_d1 = dict(fixed) if fixed else {}
        fixed_d1[var1] = 0.0
        _, _, Z_d2_only = self.predict_grid(var1, var2, n_points, fixed_d1)

        fixed_d2 = dict(fixed) if fixed else {}
        fixed_d2[var2] = 0.0
        _, _, Z_d1_only = self.predict_grid(var1, var2, n_points, fixed_d2)

        if method == "hsa":
            expected = np.minimum(Z_d1_only, Z_d2_only)
        else:  # bliss
            # Bliss: E_expected = E_d1 * E_d2 (for fractional viability)
            expected = (Z_d1_only / 100.0) * (Z_d2_only / 100.0) * 100.0

        synergy = Z_combo - expected  # negative = synergistic (lower than expected)
        return arr1, arr2, Z_combo, synergy


# ═════════════════════════════════════════════════════════════════════════════
#  BRAIDSurface — wraps BRAID parameters (from lookup or NN prediction)
# ═════════════════════════════════════════════════════════════════════════════

class BRAIDSurface(Surface):
    """
    Surface from BRAID equation parameters.

    Constructed from either:
      - braid_lookup.query_surface() result dict
      - braid_predict.BRAIDPredictionEngine.predict_surface() result dict
    """

    def __init__(self, params, drug1_name="Drug 1", drug2_name="Drug 2",
                 source="BRAID", dose_max_mult=4.0):
        """
        Args:
            params: dict with keys EC50_1, h1, EC50_2, h2, E0, Einf, kappa
                    and optionally drug1, drug2, cell_line, source, r2_braid, rmse_braid
            drug1_name, drug2_name: display names (overridden by params if present)
            source: label for params_table
            dose_max_mult: dose range = [0, dose_max_mult * EC50]
        """
        self._p = params
        self._d1 = params.get("drug1", drug1_name)
        self._d2 = params.get("drug2", drug2_name)
        self._source = params.get("source", source)
        self._mult = dose_max_mult

    @property
    def var_names(self):
        return [self._d1, self._d2]

    @property
    def response_name(self):
        return "Viability (%)"

    @property
    def bounds(self):
        return {
            self._d1: (0.0, self._mult * self._p["EC50_1"]),
            self._d2: (0.0, self._mult * self._p["EC50_2"]),
        }

    @property
    def params_table(self):
        rows = [
            ("EC50_1", f"{self._p['EC50_1']:.4g}"),
            ("h1 (Hill slope)", f"{self._p['h1']:.3f}"),
            ("EC50_2", f"{self._p['EC50_2']:.4g}"),
            ("h2 (Hill slope)", f"{self._p['h2']:.3f}"),
            ("E0 (baseline %)", f"{self._p['E0']:.1f}"),
            ("Einf (max effect %)", f"{self._p['Einf']:.1f}"),
            ("kappa", f"{self._p['kappa']:.3f}"),
        ]
        if "r2_braid" in self._p:
            rows.append(("R²", f"{self._p['r2_braid']:.3f}"))
        if "rmse_braid" in self._p:
            rows.append(("RMSE", f"{self._p['rmse_braid']:.2f}"))
        return rows

    @property
    def source_label(self):
        return self._source

    @property
    def kappa(self):
        return self._p["kappa"]

    @property
    def cell_line(self):
        return self._p.get("cell_line", "Unknown")

    @property
    def raw_params(self):
        """Access underlying parameter dict."""
        return self._p

    def predict(self, df):
        d1_col = self._d1
        d2_col = self._d2
        D1 = df[d1_col].values.astype(float)
        D2 = df[d2_col].values.astype(float)

        EC1, EC2 = self._p["EC50_1"], self._p["EC50_2"]
        h1, h2 = self._p["h1"], self._p["h2"]
        E0, Einf = self._p["E0"], self._p["Einf"]
        kappa = self._p["kappa"]

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            T1 = np.where(D1 > 0, np.clip((D1 / EC1) ** h1, 0, 1e6), 0.0)
            T2 = np.where(D2 > 0, np.clip((D2 / EC2) ** h2, 0, 1e6), 0.0)
            denom = np.clip(1.0 + T1 + T2 + kappa * T1 * T2, 1e-9, None)
            V = Einf + (E0 - Einf) / denom

        return V

    def summary(self):
        from logic.braid_lookup import kappa_label
        label, _ = kappa_label(self._p["kappa"])
        lines = [
            f"BRAID Surface: {self._d1} + {self._d2}",
            f"  Cell line: {self._p.get('cell_line', 'N/A')}",
            f"  Source: {self._source}",
            f"  kappa = {self._p['kappa']:.3f} ({label})",
            f"  E0 = {self._p['E0']:.1f}%, Einf = {self._p['Einf']:.1f}%",
            f"  EC50_1 = {self._p['EC50_1']:.4g}, h1 = {self._p['h1']:.3f}",
            f"  EC50_2 = {self._p['EC50_2']:.4g}, h2 = {self._p['h2']:.3f}",
        ]
        if "r2_braid" in self._p:
            lines.append(f"  R² = {self._p['r2_braid']:.3f}")
        return "\n".join(lines)

    def optimize_braid(self, objective="min_dose", target_viability=50.0,
                       d1_max_mult=4.0, d2_max_mult=4.0, dose_weight=(1.0, 1.0)):
        """BRAID-specific optimization (delegates to braid_lookup.optimize_braid_dose)."""
        from logic.braid_lookup import optimize_braid_dose
        return optimize_braid_dose(
            self._p, objective=objective,
            target_viability=target_viability,
            d1_max_mult=d1_max_mult, d2_max_mult=d2_max_mult,
            dose_weight=dose_weight)


# ═════════════════════════════════════════════════════════════════════════════
#  ModelSurface — adapter wrapping OLS/SVR/RF model wrappers
# ═════════════════════════════════════════════════════════════════════════════

class ModelSurface(Surface):
    """
    Surface from a fitted sklearn/statsmodels wrapper.

    Wraps the existing OLSWrapper, SVRWrapper, or RandomForestWrapper
    from logic/models.py into the unified Surface interface.
    """

    def __init__(self, wrapped_model, dep_var_name, dataframe,
                 variable_descriptions=None):
        """
        Args:
            wrapped_model: OLSWrapper / SVRWrapper / RandomForestWrapper instance
            dep_var_name: name of the dependent variable (response)
            dataframe: the expanded dataframe used for fitting (for extracting bounds)
            variable_descriptions: optional {var: description} mapping
        """
        self._model = wrapped_model
        self._dep_var = dep_var_name
        self._desc = variable_descriptions or {}

        # Extract independent vars (base variables, no polynomial terms)
        self._vars = list(wrapped_model.independent_vars)

        # Compute bounds from data
        self._bounds = {}
        for v in self._vars:
            if v in dataframe.columns:
                col = dataframe[v].dropna()
                self._bounds[v] = (float(col.min()), float(col.max()))
            else:
                self._bounds[v] = (0.0, 1.0)

    @property
    def var_names(self):
        return self._vars

    @property
    def response_name(self):
        desc = self._desc.get(self._dep_var, self._dep_var)
        return desc

    @property
    def bounds(self):
        return dict(self._bounds)

    def set_bounds(self, var, low, high):
        """Override bounds for a variable (from user input)."""
        self._bounds[var] = (float(low), float(high))

    @property
    def params_table(self):
        rows = [("Response", self._dep_var)]
        model_type = type(self._model).__name__.replace("Wrapper", "")
        rows.append(("Model", model_type))
        if hasattr(self._model, "formula"):
            rows.append(("Formula", str(self._model.formula)[:80]))
        if hasattr(self._model, "r2_score") and self._model.r2_score is not None:
            rows.append(("R²", f"{self._model.r2_score:.4f}"))
        return rows

    @property
    def source_label(self):
        return type(self._model).__name__.replace("Wrapper", "") + " fit"

    @property
    def wrapped_model(self):
        """Access underlying model wrapper (for functions that need it directly)."""
        return self._model

    def predict(self, df):
        # Ensure only base vars are present + let wrapper handle polynomial expansion
        pred = self._model.predict(df)
        return np.asarray(pred).flatten()

    def summary(self):
        return self._model.get_summary()


# ═════════════════════════════════════════════════════════════════════════════
#  Factory helpers
# ═════════════════════════════════════════════════════════════════════════════

def surface_from_braid_lookup(lookup_result, **kwargs):
    """Create a BRAIDSurface from a braid_lookup.query_surface() result."""
    return BRAIDSurface(lookup_result, source="BRAID lookup (fitted)", **kwargs)


def surface_from_braid_prediction(predict_result, checkpoint_name="v19", **kwargs):
    """Create a BRAIDSurface from a braid_predict prediction result."""
    return BRAIDSurface(predict_result, source=f"v19 NN ({checkpoint_name})", **kwargs)


def surface_from_session(dep_var_name, session_state):
    """
    Create a ModelSurface from current Streamlit session state.

    Looks up wrapped_models[dep_var_name] and expanded_df.
    Returns None if not available.
    """
    models = session_state.get("wrapped_models", {})
    df = session_state.get("expanded_df")
    desc = session_state.get("variable_descriptions", {})
    if dep_var_name not in models or df is None:
        return None
    return ModelSurface(models[dep_var_name], dep_var_name, df, desc)


def surfaces_from_session(session_state):
    """Create ModelSurface for every model in session. Returns dict {dep_var: ModelSurface}."""
    models = session_state.get("wrapped_models", {})
    df = session_state.get("expanded_df")
    desc = session_state.get("variable_descriptions", {})
    if not models or df is None:
        return {}
    return {
        name: ModelSurface(m, name, df, desc)
        for name, m in models.items()
    }
