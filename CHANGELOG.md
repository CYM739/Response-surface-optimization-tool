# Changelog

All notable changes to the Response Surface Optimization Tool will be documented in this file.

## [Unreleased] - 2026-03-04

### Fixed (Code Review — 2026-03-21)
- **FutureWarning in bootstrap diagnostics**: `original_params[i]` in `diagnostics.py` was using integer key access on a pandas Series, which pandas now treats as label-based (deprecated positional). Changed to `original_params.iloc[i]` for explicit positional access.
- **Combination Analysis returning negative outcomes**: The AI Optimizer's Combination Analysis had no guard against the OLS equation producing physically impossible negative results (e.g. negative cell viability). Added a user-configurable "Minimum acceptable outcome" input (default `0.0`) that is enforced via a penalty term in the Bayesian objective function, mirroring the `r_min` constraint in the normal optimizer.
- **Combination Analysis returning zero-dose combinations**: The Bayesian optimizer (`gp_minimize`) could return a zero dosage for one of the combination variables despite an existing penalty, because the GP surrogate smooths out sharp discontinuities and ignores the spike at exactly `x=0`. Fixed by hard-excluding zero from the search space: when a variable's minimum dose is 0, the lower bound is now set to `second_min_val` (the lowest non-zero dose actually tested in the experiment), making the constraint enforceable at the bounds level rather than via a penalty.

### Changed (Code Review — 2026-03-21)
- **OLS Diagnostics no longer requires a model dropdown**: The diagnostics view previously required the user to select one OLS model from a dropdown before seeing any results. The selectbox has been removed; all available OLS models are now rendered simultaneously, each in its own collapsible expander with independent tabs and widget keys.
- **`plot_combination_ranking` removed from `optimization.py`**: The function was called inside `run_combination_optimization` but never imported and its return value was never used (the view already calls it independently). The dead call has been removed.

### Fixed (Code Review — 2026-03-04)
- **Duplicate dead-code function removed**: `plot_synergy_heatmap` was silently defined twice in `plotting.py`. The inferior second definition (`px.imshow` version, lacking categorical axis fix and custom hovertemplate) has been deleted. The first, correct definition is now the only one.
- **Windows parallel optimizer crash/hang**: Both `concurrent.futures.ProcessPoolExecutor` calls in `optimization.py` replaced with `ThreadPoolExecutor`. On Windows, `ProcessPoolExecutor` spawned fresh processes that re-imported all heavy dependencies (sklearn, statsmodels, skopt) per worker, causing RAM spikes and potential Streamlit hangs. Threads share the process memory and numpy/scipy/sklearn release the GIL in C extensions, preserving meaningful parallelism.
- **`multiprocessing.freeze_support()` added to `app_edu.py`**: Matches defensive practice already present in `app.py`.
- **O(N²) performance in high-throughput synergy screening**: `calculate_high_throughput_synergy` in `data_processing.py` previously ran a full boolean mask scan over the entire dataframe for every active drug in every row. A lookup dict (`single_agent_cache`) is now pre-computed once before the row loop, reducing inner lookups from O(N) mask scans to O(1) dict access. Datasets with 200+ rows will no longer freeze during synergy analysis.

### Changed (Code Review — 2026-03-04)
- **SVR and Random Forest wrappers no longer store training data**: `SVRWrapper` and `RandomForestWrapper` previously kept `X_train` and `y_train` as full DataFrame slices in `session_state` (one copy per fitted model) solely to compute R² in `get_summary()`. Both wrappers now compute and store `r2_score` (a single float) immediately after fitting and the training data references are dropped.
- **Orphan imports removed from `models.py`**: Six import lines that belonged to `optimization.py` were loaded by `models.py` unnecessarily (`scipy.optimize`, `skopt`, `itertools`, `concurrent.futures`). Removed. Duplicate `import io` (appeared on lines 16 and 26) also deduplicated.
- **`_format_fixed_vars` helper extracted in `plotting.py`**: The one-liner `", ".join([f"{variable_descriptions.get(k, k)}={v}" ...])` was duplicated verbatim in three plot functions. Extracted into a module-private helper `_format_fixed_vars(fixed_vars_dict, variable_descriptions)`.
- **Shared grid-generation utility added**: `generate_prediction_grid(model, x_var, y_var, fixed_vars, x_range, y_range, n=100)` added to `helpers.py`. Meshgrid creation, flattening, and DataFrame assembly were independently duplicated in `predict_surface` (`data_processing.py`) and `calculate_synergy_surface_grid` (`plotting.py`). Both now delegate to this utility.
- **Optimizer state key list centralised**: `_OPTIMIZER_RESULT_KEYS` constant added to `state_management.py`. `clear_optimizer_results()` now iterates it instead of naming each key explicitly — adding a new optimizer now requires only one list entry.



### Added
- **Drug Elimination Score**: A new analysis tab (`🗑️ Drug Elimination Score`) that mathematically ranks drugs indicating which should be eliminated during experiments.
  - Compares the toxicity of a chosen normal cell OLS model against the efficacy of chosen cancer cell OLS models.
  - Automatically parses interaction coefficients (synergies/antagonisms) from standard OLS matrices.
  - Renders exact mathematical scoring parameters and linear OLS equation structures in the UI via LaTeX dynamically based on selected models.
  - Implemented configurable weighting parameters (Toxicity Weight & Efficacy Weight) for tunable trade-off evaluation via Streamlit sliders.

### Fixed
- **Startup Script Environments**: Fixed `start.bat` and `start_edu.bat` which were previously trying to access non-existent offline packages and Python-embed distributions. Start scripts now faithfully set up the `venv` and pull dependencies directly via pip.

### Changed
- **Deprecation Updates for Pandals & Streamlit Compatibility**: 
  - Overhauled exactly 38 instances of `use_container_width=True` to standard `width="stretch"` across the `views` directory to prepare for Streamlit's removal of legacy APIs scheduled for Dec 2025.
  - Replaced legacy Pandas `Styler.applymap()` calls with `Styler.map()` inside `diagnostics_view.py`. 
  - Force-saved all `.py` files inside the `src/` directory explicitly with UTF-8 encoding. This ensures that emojis and technical unicode symbols (e.g. `β`) do not crash the app backend when running in natively localized encodings (e.g., CP1252) on Windows.
