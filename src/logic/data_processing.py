# src/logic/data_processing.py
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from docx import Document
import io
from datetime import datetime
from .models import OLSWrapper, SVRWrapper, RandomForestWrapper
from .helpers import _add_polynomial_terms

# --- STANDARD DATA ANALYSIS FUNCTIONS ---

def analyze_csv(full_dataframe):
    """
    Inspects the input dataframe to prepare it for analysis.
    """
    first_row = full_dataframe.iloc[0]
    is_description_row = any(isinstance(item, str) for item in first_row)

    all_vars = full_dataframe.columns.tolist()

    if is_description_row:
        descriptions_row = first_row
        data_df = full_dataframe.iloc[1:].reset_index(drop=True)
        variable_descriptions = {var: descriptions_row[var] for var in all_vars}
    else:
        data_df = full_dataframe.copy()
        variable_descriptions = {var: var for var in all_vars}

    # Clean numeric columns
    for col in data_df.columns:
        if data_df[col].dtype == 'object':
            try:
                cleaned_col = data_df[col].str.replace(',', '', regex=False)
                data_df[col] = pd.to_numeric(cleaned_col)
            except (AttributeError, ValueError, TypeError):
                pass

    ignore_prefixes = ["Cell_", "Zebrafish_", "Mouse_", "Patient_", "Control_"]
    independent_vars = [
        col for col in all_vars
        if col[0].isalpha() and not any(col.startswith(prefix) for prefix in ignore_prefixes)
    ]
    dependent_vars = [
        col for col in all_vars
        if any(col.startswith(prefix) for prefix in ignore_prefixes)
    ]

    variable_stats = {}
    unique_variable_values = {}
    detected_binary_vars = []

    for var in all_vars:
        numeric_col = pd.to_numeric(data_df[var], errors='coerce').dropna()
        if not numeric_col.empty:
            unique_sorted = np.sort(numeric_col.unique())
            unique_variable_values[var] = unique_sorted.tolist()
            variable_stats[var] = (unique_sorted[0], unique_sorted[1] if len(unique_sorted) > 1 else unique_sorted[0], unique_sorted[-1])

            if var in independent_vars:
                if set(unique_sorted) == {0.0, 1.0} or set(unique_sorted) == {0, 1}:
                    detected_binary_vars.append(var)

    special_values_map = {}
    for col_name in all_vars:
        numeric_series = pd.to_numeric(data_df[col_name], errors='coerce')
        non_numeric_mask = numeric_series.isna() & data_df[col_name].notna()
        if non_numeric_mask.any():
            special_values = data_df[col_name][non_numeric_mask].unique().tolist()
            special_values_map[col_name] = special_values

    return (data_df, all_vars, independent_vars, dependent_vars,
            variable_stats, special_values_map, unique_variable_values,
            variable_descriptions, detected_binary_vars)

def expand_terms(dataframe, all_alphabet_vars):
    """Wrapper for polynomial term generation."""
    _add_polynomial_terms(dataframe, all_alphabet_vars)

def generate_model_formula(C_col, all_alphabet_vars):
    """Constructs model formula for statsmodels."""
    terms = []
    for header in all_alphabet_vars:
        terms.extend([header, f"{header}_sq"])

    if len(all_alphabet_vars) > 1:
        interaction_terms = [f"{var1}*{var2}" for i, var1 in enumerate(all_alphabet_vars) for var2 in all_alphabet_vars[i + 1:]]
        terms.extend(interaction_terms)

    return f"{C_col} ~ " + " + ".join(terms)

def run_analysis(dataframe, independent_vars, dependent_var, model_type, model_params={}, variable_descriptions=None):
    """Main analysis engine to fit models."""
    if model_type == 'Polynomial OLS':
        model_formula = generate_model_formula(dependent_var, independent_vars)
        ols_model = smf.ols(model_formula, data=dataframe).fit()
        return OLSWrapper(ols_model, model_formula)

    elif model_type == 'SVR':
        svr_wrapper = SVRWrapper(independent_vars)
        svr_wrapper.fit(dataframe, dependent_var, **model_params)
        return svr_wrapper

    elif model_type == 'Random Forest':
        rf_wrapper = RandomForestWrapper(independent_vars)
        model_params['variable_descriptions'] = variable_descriptions
        rf_wrapper.fit(dataframe, dependent_var, **model_params)
        return rf_wrapper
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# --- NEW SYNERGY / HIGH-THROUGHPUT FUNCTIONS ---

def detect_effect_direction(df, dose_cols, effect_col):
    """
    Automatically detects if the 'effect' is Inhibition (Higher=Stronger) 
    or Viability (Lower=Stronger) based on correlation.
    
    Returns:
        direction (str): 'inhibition' or 'viability'
        is_transformed (bool): True if logic decided to flip the data.
    """
    # Calculate simple correlation between sum of doses and effect
    total_dose = df[dose_cols].sum(axis=1)
    corr = total_dose.corr(df[effect_col])
    
    # If correlation is Positive (>0), Higher Dose = Higher Value -> Inhibition
    # If correlation is Negative (<0), Higher Dose = Lower Value -> Viability
    if corr < 0:
        return 'viability'
    else:
        return 'inhibition'

def get_single_agent_response(row_idx, current_row, drug_name, df_clean, dose_cols, effect_col, model=None):
    """
    Finds the single-agent effect for 'drug_name' at the dose specified in 'current_row'.
    1. Looks for exact experimental match in df_clean.
    2. If missing, uses 'model' to predict it.
    """
    target_dose = current_row[drug_name]
    
    # If dose is effectively 0, effect is 0 (relative to baseline)
    if target_dose < 1e-6:
        return 0.0

    # 1. Search for experimental data
    # Logic: Find rows where 'drug_name' is close to target_dose AND all other drugs are 0
    mask = (np.abs(df_clean[drug_name] - target_dose) < 1e-6)
    for other_drug in dose_cols:
        if other_drug != drug_name:
            mask &= (np.abs(df_clean[other_drug]) < 1e-6)
    
    matches = df_clean[mask]
    
    if not matches.empty:
        # Return mean of replicates
        return matches[effect_col].mean()
    
    # 2. Fallback: AI Prediction
    if model is not None:
        # Construct a synthetic row for prediction
        pred_input = {col: 0.0 for col in dose_cols}
        pred_input[drug_name] = target_dose
        pred_df = pd.DataFrame([pred_input])
        
        # Predict
        try:
            prediction = model.predict(pred_df)[0]
            # Clip prediction to reasonable bounds (0-1 for inhibition)
            return np.clip(prediction, 0.0, 1.0) 
        except:
            return np.nan
            
    return np.nan

def calculate_high_throughput_synergy(df, dose_cols, effect_col, model=None):
    """
    Row-based synergy calculation supporting N-drugs, missing data (AI-fill), 
    and Bliss/HSA scores.
    """
    results = df.copy()
    
    # 1. Auto-Detect Direction & Standardize to Inhibition (0=No Effect, 1=Full Kill)
    direction = detect_effect_direction(df, dose_cols, effect_col)
    
    # Create an internal working column 'std_effect' (Standardized Effect)
    if direction == 'viability':
        # Assume 0-1 range. Transform: Inhibition = 1 - Viability
        # Check max value to be safe
        max_val = df[effect_col].max()
        if max_val > 1.5: # Likely percentage 0-100
             results['std_effect'] = 1 - (df[effect_col] / 100.0)
        else:
             results['std_effect'] = 1 - df[effect_col]
    else:
        # Already inhibition. Check bounds.
        if df[effect_col].max() > 1.5:
            results['std_effect'] = df[effect_col] / 100.0
        else:
            results['std_effect'] = df[effect_col]
            
    # Lists to store results
    hsa_synergy_scores = []
    bliss_synergy_scores = []
    notes = []
    
    # Pre-clean for lookups
    df_lookup = results.copy()
    for col in dose_cols:
        df_lookup[col] = pd.to_numeric(df_lookup[col], errors='coerce').fillna(0)
    
    # 2. Iterate Rows
    for idx, row in df_lookup.iterrows():
        # Identify Active Drugs in this row (Dose > 0)
        active_drugs = [d for d in dose_cols if row[d] > 1e-6]
        
        if len(active_drugs) < 2:
            # Not a combination
            hsa_synergy_scores.append(0.0)
            bliss_synergy_scores.append(0.0)
            notes.append("Single Agent / Control")
            continue
            
        # Get Single Agent Effects (Observed or Predicted)
        single_agent_effects = []
        is_predicted = False
        
        for drug in active_drugs:
            eff = get_single_agent_response(idx, row, drug, df_lookup, dose_cols, 'std_effect', model)
            if pd.isna(eff):
                is_predicted = True # Flag error actually
            single_agent_effects.append(eff)
            
        if any(pd.isna(x) for x in single_agent_effects):
            hsa_synergy_scores.append(np.nan)
            bliss_synergy_scores.append(np.nan)
            notes.append("Missing Reference Data")
            continue
            
        # 3. Calculate HSA (Highest Single Agent)
        # Exp_HSA = Max(Single Agent Effects)
        # Synergy = Obs - Exp
        obs_effect = row['std_effect']
        exp_hsa = max(single_agent_effects)
        hsa_score = obs_effect - exp_hsa
        
        # 4. Calculate Bliss (Independence)
        # Probabilistic Independence: P(A+B) = P(A) + P(B) - P(A)*P(B)
        # Or easier: Unaffected_Exp = (1-E_A) * (1-E_B) ...
        # Exp_Bliss = 1 - Unaffected_Exp
        unaffected_product = 1.0
        for eff in single_agent_effects:
            unaffected_product *= (1.0 - eff)
        
        exp_bliss = 1.0 - unaffected_product
        bliss_score = obs_effect - exp_bliss
        
        hsa_synergy_scores.append(hsa_score)
        bliss_synergy_scores.append(bliss_score)
        notes.append("Calculated" if not is_predicted else "AI Estimated Ref")

    # 5. Append Results
    results['HSA_Synergy'] = hsa_synergy_scores
    results['Bliss_Synergy'] = bliss_synergy_scores
    results['Analysis_Note'] = notes
    
    return results, direction
