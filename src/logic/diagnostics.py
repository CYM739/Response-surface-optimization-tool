# src/logic/diagnostics.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.tools import add_constant 
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
import io
from .models import OLSWrapper
from .helpers import _add_polynomial_terms 

def calculate_vif(model_wrapper, dataframe=None, independent_vars=None):
    """
    Calculates Variance Inflation Factor (VIF).
    
    If 'dataframe' and 'independent_vars' are provided, this function 
    PERFORMS MEAN CENTERING on the data before calculating VIF.
    This removes 'Structural Multicollinearity' (caused by x and x^2) 
    to show the true underlying correlation between drugs.
    """
    # A. If data is provided, build a CENTERED Design Matrix (The Fix)
    if dataframe is not None and independent_vars is not None:
        # 1. Copy and Center the continuous variables
        df_centered = dataframe.copy()
        for var in independent_vars:
            if pd.api.types.is_numeric_dtype(df_centered[var]):
                mean_val = df_centered[var].mean()
                df_centered[var] = df_centered[var] - mean_val
        
        # 2. Re-create polynomial terms (Squared, Interaction) using centered data
        _add_polynomial_terms(df_centered, independent_vars)
        
        # 3. Create the Design Matrix
        # Filter: Take independent vars + generated poly terms
        # We look for cols that match independent vars OR contain '_sq' / '*'
        target_cols = independent_vars + [c for c in df_centered.columns if '_sq' in c or '*' in c]
        
        # Ensure we only use columns that exist
        target_cols = [c for c in target_cols if c in df_centered.columns]
        
        X = df_centered[target_cols].dropna()
        X = add_constant(X)
    
    # B. Default: Use the existing Design Matrix from the fitted model (Uncentered)
    else:
        X = pd.DataFrame(
            model_wrapper.model.model.exog, 
            columns=model_wrapper.model.model.exog_names
        )

    # Calculate VIF
    vif_data = []
    for i, name in enumerate(X.columns):
        if name.lower() == 'const' or name.lower() == 'intercept':
            continue
            
        try:
            # We use X.values to ensure compatibility
            val = variance_inflation_factor(X.values, i)
            vif_data.append({'Feature': name, 'VIF': val})
        except Exception:
            vif_data.append({'Feature': name, 'VIF': np.nan})
        
    return pd.DataFrame(vif_data)

def perform_normality_test(residuals):
    """
    Performs Shapiro-Wilk test for normality of residuals.
    """
    if len(residuals) > 5000:
        shapiro_stat, shapiro_p = stats.jarque_bera(residuals)
        test_name = "Jarque-Bera"
    else:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        test_name = "Shapiro-Wilk"
        
    return shapiro_stat, shapiro_p, test_name

def perform_heteroscedasticity_test(residuals, ols_model_wrapper):
    """
    Performs Breusch-Pagan test for heteroscedasticity.
    """
    exog = ols_model_wrapper.model.model.exog
    lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residuals, exog)
    return lm_p_value

def perform_autocorrelation_test(residuals):
    """
    Performs Durbin-Watson test for autocorrelation.
    """
    dw_stat = durbin_watson(residuals)
    return dw_stat

# --- NEW FUNCTIONS FOR UNCERTAINTY QUANTIFICATION ---

def perform_kfold_cv(model_wrapper, k=5):
    """
    Performs K-Fold Cross-Validation to estimate out-of-sample performance.
    """
    results = model_wrapper.model
    # statsmodels stores the design matrix in results.model.exog/endog
    X = results.model.exog
    y = results.model.endog
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    cv_scores = {'mse': [], 'rmse': [], 'r2': []}
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Refit OLS on training fold
        model_fold = sm.OLS(y_train, X_train).fit()
        preds = model_fold.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        cv_scores['mse'].append(mse)
        cv_scores['rmse'].append(np.sqrt(mse))
        cv_scores['r2'].append(r2_score(y_test, preds))
        
    return {
        'k': k,
        'avg_mse': np.mean(cv_scores['mse']),
        'std_mse': np.std(cv_scores['mse']),
        'avg_rmse': np.mean(cv_scores['rmse']),
        'std_rmse': np.std(cv_scores['rmse']),
        'avg_r2': np.mean(cv_scores['r2']),
        'std_r2': np.std(cv_scores['r2'])
    }

def perform_bootstrap_analysis(model_wrapper, n_bootstraps=100):
    """
    Performs Bootstrap resampling to estimate coefficient stability and confidence intervals.
    """
    results = model_wrapper.model
    X = results.model.exog
    y = results.model.endog
    exog_names = results.model.exog_names
    original_params = results.params
    
    boot_params = []
    
    for _ in range(n_bootstraps):
        # Resample with replacement
        X_res, y_res = resample(X, y, random_state=None)
        try:
            model_boot = sm.OLS(y_res, X_res).fit()
            boot_params.append(model_boot.params)
        except:
            continue # Skip failed fits
            
    boot_df = pd.DataFrame(boot_params)
    
    stats_list = []
    # Columns in boot_df usually correspond to indices of exog_names
    for i, name in enumerate(exog_names):
        if i >= boot_df.shape[1]: continue
        
        values = boot_df.iloc[:, i]
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        orig = original_params.iloc[i] if i < len(original_params) else 0
        
        # FIX: Robust check for stability
        # Stable if 0 is NOT between lower and upper
        # Meaning: Both are positive OR Both are negative
        if lower > 0 or upper < 0:
            stable_icon = "✅"
        else:
            # 0 is inside the interval (lower <= 0 <= upper)
            stable_icon = "⚠️"
        
        stats_list.append({
            'Term': name,
            'Original': orig,
            'Boot Mean': values.mean(),
            '95% CI Lower': lower,
            '95% CI Upper': upper,
            'Stable?': stable_icon
        })
        
    return pd.DataFrame(stats_list)

def generate_diagnostics_report(model_wrapper, model_name="Model", dataframe=None, independent_vars=None):
    """Aggregates diagnostic tests, now supports Centered VIF if data is passed."""
    results = model_wrapper.model
    residuals = results.resid
    report = []
    report.append(f"Model Name: {model_name}\nFormula: {model_wrapper.formula}\n" + "-" * 50)
    
    report.append("1. MULTICOLLINEARITY (Variance Inflation Factor)")
    try:
        # Pass the data to calculate_vif so it can perform centering
        vif_df = calculate_vif(model_wrapper, dataframe=dataframe, independent_vars=independent_vars)
        if dataframe is not None:
             report.append("(NOTE: VIFs are Mean-Centered to remove structural multicollinearity.)")
        report.append(vif_df.to_string(index=False, float_format="{:.4f}".format))
    except Exception as e: report.append(f"Error: {e}")
    
    report.append("\n2. NORMALITY")
    stat, p_val, test_name = perform_normality_test(residuals)
    report.append(f"{test_name} p-value: {p_val:.4f} -> " + ("FAIL" if p_val < 0.05 else "PASS"))
    
    report.append("\n3. HOMOSCEDASTICITY")
    lm_p = perform_heteroscedasticity_test(residuals, model_wrapper)
    report.append(f"Breusch-Pagan p-value: {lm_p:.4f} -> " + ("FAIL" if lm_p < 0.05 else "PASS"))
    
    report.append("\n4. INDEPENDENCE")
    dw_stat = perform_autocorrelation_test(residuals)
    report.append(f"Durbin-Watson: {dw_stat:.4f}")
    
    report.append("\n5. PREDICTIVE UNCERTAINTY")
    try:
        cv = perform_kfold_cv(model_wrapper)
        report.append(f"CV Avg RMSE: {cv['avg_rmse']:.4f}")
    except: pass
    
    return "\n".join(report)

def generate_full_project_report(wrapped_models, dataframe=None, independent_vars=None):
    """Iterates through all OLS models, passing data down for centering."""
    full = ["=== PROJECT DIAGNOSTICS REPORT ===\n"]
    for name, model in wrapped_models.items():
        if isinstance(model, OLSWrapper):
            # Pass the data down to the single report generator
            full.append(generate_diagnostics_report(model, name, dataframe=dataframe, independent_vars=independent_vars))
            full.append("\n" + "="*30 + "\n")
    return "\n".join(full)
