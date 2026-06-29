"""
Shared pytest configuration for the AI-PRS test suite.

Puts `src/` on sys.path so tests can use the same `from logic.x import y` /
`from views import y` style imports that the Streamlit app uses at runtime.
Also provides reusable fixtures (paths, analyzed demo data, a trained OLS
model) so each test stays small.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Make the Streamlit app resolve relative paths (config.json, project_library.json)
# from the project root, matching what start.bat does.
os.chdir(PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def demo_csv_path(project_root) -> Path:
    path = project_root / "demo_3drug_dri.csv"
    assert path.exists(), f"Expected demo CSV at {path}"
    return path


@pytest.fixture(scope="session")
def demo_df(demo_csv_path) -> pd.DataFrame:
    return pd.read_csv(demo_csv_path)


@pytest.fixture(scope="session")
def analyzed_demo(demo_df):
    """Run the app's CSV inspection pipeline on the demo data.

    Returns a dict with the same keys the app stores in st.session_state after
    a CSV is loaded via the Project Library tab.
    """
    from logic.data_processing import analyze_csv

    (
        data_df,
        all_vars,
        independent_vars,
        dependent_vars,
        variable_stats,
        special_values_map,
        unique_variable_values,
        variable_descriptions,
        detected_binary_vars,
    ) = analyze_csv(demo_df.copy())

    return {
        "exp_df": data_df,
        "all_vars": all_vars,
        "independent_vars": independent_vars,
        "dependent_vars": dependent_vars,
        "variable_stats": variable_stats,
        "special_values_map": special_values_map,
        "unique_variable_values": unique_variable_values,
        "variable_descriptions": variable_descriptions,
        "detected_binary_vars": detected_binary_vars,
    }


@pytest.fixture(scope="session")
def trained_ols(analyzed_demo):
    """Fit a Polynomial OLS model on the demo data's first dependent variable.

    Mirrors the call chain library_view.py runs when the user clicks
    'Run and Save Analysis': expand_terms() first, then run_analysis().
    """
    from logic.data_processing import expand_terms, run_analysis

    dep_var = analyzed_demo["dependent_vars"][0]
    df_for_analysis = analyzed_demo["exp_df"].copy()
    expand_terms(df_for_analysis, analyzed_demo["independent_vars"])

    wrapper = run_analysis(
        dataframe=df_for_analysis,
        independent_vars=analyzed_demo["independent_vars"],
        dependent_var=dep_var,
        model_type="Polynomial OLS",
        variable_descriptions=analyzed_demo["variable_descriptions"],
    )
    return {"dep_var": dep_var, "wrapper": wrapper, "expanded_df": df_for_analysis}
