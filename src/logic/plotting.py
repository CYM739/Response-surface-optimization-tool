# src/logic/plotting.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from .data_processing import predict_surface, expand_terms
from .optimization import objective_function
from .models import OLSWrapper
from .helpers import _add_polynomial_terms

def plot_response_surface(dataframe, OLS_model_1, all_alphabet_vars, x_var, y_var,
                          z_var_1, fixed_vars_dict_1, variable_descriptions,
                          OLS_model_2=None, z_var_2=None, fixed_vars_dict_2=None,
                          optimized_point=None, optimized_point_2=None, 
                          point_size=8, show_actual_data=True,
                          main_title=None, x_title=None, y_title=None, z_title=None,
                          colorscale_1='Plasma', colorscale_2='Greys',
                          show_x_grid=True, show_y_grid=True, show_surface_grid=True, 
                          title_font_size=18, axis_title_font_size=12, axis_tick_font_size=10,
                          point_symbol_1='x', point_color_1='red',
                          point_symbol_2='cross', point_color_2='#00FF00', z_range=None, **kwargs):
    
    x_min, x_max = dataframe[x_var].min(), dataframe[x_var].max()
    y_min, y_max = dataframe[y_var].min(), dataframe[y_var].max()
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)

    x_grid_1, y_grid_1, z_grid_1 = predict_surface(OLS_model_1, all_alphabet_vars, x_var, y_var, fixed_vars_dict_1, x_range, y_range)

    # --- NEW LOGIC: Dynamic Grid Density ---
    contours_config = None
    if show_surface_grid:
        # Calculate step size to ensure roughly 15 grid lines per axis
        # This fixes the "big squares" issue
        grid_density = 15
        x_step = (x_max - x_min) / grid_density
        y_step = (y_max - y_min) / grid_density

        contours_config = {
            "x": {
                "show": True, "start": x_min, "end": x_max, "size": x_step,
                "color": "black", "width": 1, "highlight": False
            },
            "y": {
                "show": True, "start": y_min, "end": y_max, "size": y_step,
                "color": "black", "width": 1, "highlight": False
            }
        }
    # ---------------------------------------

    trace1 = go.Surface(
        z=z_grid_1, x=x_grid_1, y=y_grid_1, 
        colorscale=colorscale_1, opacity=0.9, 
        name=f'Surface: {z_var_1}',
        contours=contours_config
    )
    
    if z_range:
        trace1.update(cmin=z_range[0], cmax=z_range[1])
    traces = [trace1]

    if OLS_model_2 and z_var_2 and fixed_vars_dict_2 is not None:
        x_grid_2, y_grid_2, z_grid_2 = predict_surface(OLS_model_2, all_alphabet_vars, x_var, y_var, fixed_vars_dict_2, x_range, y_range)
        trace2 = go.Surface(
            z=z_grid_2, x=x_grid_2, y=y_grid_2, 
            colorscale=colorscale_2, opacity=0.6, 
            name=f'Surface: {z_var_2}', showscale=False,
            contours=contours_config
        )
        if z_range:
            trace2.update(cmin=z_range[0], cmax=z_range[1])
        traces.append(trace2)

    if show_actual_data:
        mask = pd.Series(True, index=dataframe.index)
        for var, val in fixed_vars_dict_1.items():
            if pd.api.types.is_numeric_dtype(dataframe[var]):
                mask &= np.isclose(dataframe[var], float(val))
        filtered_df = dataframe[mask]
        if not filtered_df.empty:
            traces.append(go.Scatter3d(x=filtered_df[x_var], y=filtered_df[y_var], z=filtered_df[z_var_1], mode='markers', marker=dict(color='red', size=4), name='Actual Data'))

    if optimized_point is not None:
        ordered_dosages = [optimized_point[var] for var in all_alphabet_vars]
        point_z_1 = objective_function(ordered_dosages, OLS_model_1, all_alphabet_vars)
        traces.append(go.Scatter3d(x=[optimized_point[x_var]], y=[optimized_point[y_var]], z=[point_z_1], mode='markers', 
                                   marker=dict(color=point_color_1, size=point_size, symbol=point_symbol_1), name='Optimal Point 1'))

    if OLS_model_2 and optimized_point_2 is not None:
        ordered_dosages_2 = [optimized_point_2[var] for var in all_alphabet_vars]
        point_z_2 = objective_function(ordered_dosages_2, OLS_model_2, all_alphabet_vars)
        traces.append(go.Scatter3d(x=[optimized_point_2[x_var]], y=[optimized_point_2[y_var]], z=[point_z_2], mode='markers',
                                   marker=dict(color=point_color_2, size=point_size, symbol=point_symbol_2), name='Optimal Point 2'))

    fig = go.Figure(data=traces)

    z_desc_1 = variable_descriptions.get(z_var_1, z_var_1)
    x_desc = variable_descriptions.get(x_var, x_var)
    y_desc = variable_descriptions.get(y_var, y_var)

    title_fixed_vars = ", ".join([f"{variable_descriptions.get(k, k)}={v}" for k, v in fixed_vars_dict_1.items()])
    default_title = f'Response Surface for {z_desc_1}<br>(Fixed: {title_fixed_vars})'
    
    scene_dict = dict(
            xaxis = dict(
                title=dict(text=x_title or x_desc, font=dict(size=axis_title_font_size)),
                tickfont=dict(size=axis_tick_font_size),
                showgrid=show_x_grid, gridcolor='lightgrey', gridwidth=1, zeroline=False
            ),
            yaxis = dict(
                title=dict(text=y_title or y_desc, font=dict(size=axis_title_font_size)),
                tickfont=dict(size=axis_tick_font_size),
                showgrid=show_y_grid, gridcolor='lightgrey', gridwidth=1, zeroline=False
            ),
            zaxis = dict(
                title=dict(text=z_title or z_desc_1, font=dict(size=axis_title_font_size)),
                tickfont=dict(size=axis_tick_font_size),
                showgrid=True, gridcolor='lightgrey', gridwidth=1, zeroline=False
            )
        )

    if z_range:
        scene_dict['zaxis']['range'] = z_range
        
    fig.update_layout(
        title=dict(text=main_title or default_title, font=dict(size=title_font_size)),
        scene = scene_dict,
        height=700
    )
    return fig

def plot_response_curve(dataframe, OLS_model, independent_var, dependent_var,
                        variable_descriptions, optimized_point=None,
                        optimized_point_name='Optimized Point',
                        main_title=None, x_title=None, y_title=None,
                        actual_color='orangered', curve_color='royalblue',
                        actual_size=8, curve_width=3, show_actual_data=True,
                        opt_point_color='red', opt_point_symbol='x', opt_point_size=15,
                        title_font_size=18, axis_title_font_size=12, axis_tick_font_size=10):
    """
    Generates an interactive 2D response curve for cases with only one independent variable.
    """
    x_range = np.linspace(dataframe[independent_var].min(), dataframe[independent_var].max(), 200)
    predict_df = pd.DataFrame({independent_var: x_range})
    if isinstance(OLS_model, OLSWrapper):
        expand_terms(predict_df, [independent_var])

    y_predicted_curve = OLS_model.predict(predict_df)

    x_actual = dataframe[independent_var]
    y_actual = pd.to_numeric(dataframe[dependent_var], errors='coerce')
    valid_indices = y_actual.notna()
    x_actual_valid = x_actual[valid_indices]
    y_actual_valid = y_actual[valid_indices]

    x_desc = variable_descriptions.get(independent_var, independent_var)
    y_desc = variable_descriptions.get(dependent_var, dependent_var)

    fig = go.Figure()

    if show_actual_data:
        fig.add_trace(go.Scatter(
            x=x_actual_valid, y=y_actual_valid, mode='markers', name='Actual Data Points',
            marker=dict(color=actual_color, size=actual_size, opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))
        ))

    fig.add_trace(go.Scatter(
        x=x_range, y=y_predicted_curve, mode='lines', name='Model Fit Curve',
        line=dict(color=curve_color, width=curve_width)
    ))

    if optimized_point is not None:
        optimized_x = optimized_point[independent_var]
        optimized_y = objective_function([optimized_x], OLS_model, [independent_var])
        fig.add_trace(go.Scatter(
            x=[optimized_x], y=[optimized_y], mode='markers', name=optimized_point_name,
            marker=dict(color=opt_point_color, size=opt_point_size, symbol=opt_point_symbol, line=dict(width=3, color='black'))
        ))

    default_title = f"Response Curve for {y_desc} vs. {x_desc}"
    fig.update_layout(
        title=dict(text=main_title if main_title else default_title, font=dict(size=title_font_size)),
        xaxis_title=dict(text=x_title if x_title else x_desc, font=dict(size=axis_title_font_size)),
        yaxis_title=dict(text=y_title if y_title else y_desc, font=dict(size=axis_title_font_size)),
        legend=dict(x=0.01, y=0.99),
        height=600
    )
    fig.update_xaxes(tickfont=dict(size=axis_tick_font_size))
    fig.update_yaxes(tickfont=dict(size=axis_tick_font_size))
    
    return fig

def plot_actual_vs_predicted(y_actual, y_predicted, dependent_var, variable_descriptions,
                             main_title=None, x_title=None, y_title=None,
                             font_size=10, bar_width=0.35, dot_size=30):
    """
    Generates two plots to evaluate model fit.
    """
    dep_var_desc = variable_descriptions.get(dependent_var, dependent_var)
    final_main_title = main_title if main_title else f'Actual vs. Predicted ({dep_var_desc})'
    final_x_title_scatter = x_title if x_title else f'Actual {dep_var_desc}'
    final_y_title_scatter = y_title if y_title else f'Predicted {dep_var_desc}'

    index = y_actual.index
    x_pos = np.arange(len(index))

    # Bar Chart
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(x_pos - bar_width/2, y_actual.values, bar_width, label='Actual', alpha=0.7)
    ax1.bar(x_pos + bar_width/2, y_predicted.values, bar_width, label='Predicted', alpha=0.7)
    ax1.set_xlabel('Experiment Index', fontsize=font_size)
    ax1.set_ylabel('Outcome Value', fontsize=font_size)
    ax1.set_title(final_main_title, fontsize=font_size + 2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(index)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='both', which='major', labelsize=font_size - 1)
    ax1.legend(fontsize=font_size - 1)
    fig1.tight_layout()

    # Scatter Plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(y_actual, y_predicted, s=dot_size, alpha=0.7)
    if not y_actual.empty and not y_predicted.empty:
        numeric_y_actual = pd.to_numeric(y_actual, errors='coerce').dropna()
        numeric_y_predicted = pd.to_numeric(y_predicted, errors='coerce').dropna()
        if not numeric_y_actual.empty and not numeric_y_predicted.empty:
            lims = [
                np.min([numeric_y_actual.min(), numeric_y_predicted.min()]),
                np.max([numeric_y_actual.max(), numeric_y_predicted.max()])
            ]
            ax2.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal Fit')

    ax2.set_xlabel(final_x_title_scatter, fontsize=font_size)
    ax2.set_ylabel(final_y_title_scatter, fontsize=font_size)
    ax2.set_title(final_main_title, fontsize=font_size + 2)
    ax2.tick_params(axis='both', which='major', labelsize=font_size - 1)
    ax2.legend(fontsize=font_size - 1)
    ax2.grid(True)
    fig2.tight_layout()

    return fig1, fig2

def plot_tradeoff_analysis(model_1, model_2, model_name_1, model_name_2, all_alphabet_vars,
                           x_var, fixed_vars_dict, dataframe, variable_descriptions):
    """
    Generates a 2D line plot showing how two different model outcomes (and their
    difference) change as a single independent variable is varied.
    """
    x_range = np.linspace(dataframe[x_var].min(), dataframe[x_var].max(), 100)
    predict_df = pd.DataFrame({x_var: x_range})
    for var, val in fixed_vars_dict.items():
        predict_df[var] = val
    if isinstance(model_1, OLSWrapper):
        _add_polynomial_terms(predict_df, all_alphabet_vars)

    pred_1 = model_1.predict(predict_df)
    pred_2 = model_2.predict(predict_df)
    difference = pred_1 - pred_2

    desc_1 = variable_descriptions.get(model_name_1, model_name_1)
    desc_2 = variable_descriptions.get(model_name_2, model_name_2)
    x_desc = variable_descriptions.get(x_var, x_var)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x_range, pred_1, label=f'Outcome: {desc_1}', color='tab:blue')
    ax1.plot(x_range, pred_2, label=f'Outcome: {desc_2}', color='tab:red')
    ax1.set_xlabel(x_desc)
    ax1.set_ylabel('Predicted Outcome Value')

    ax2 = ax1.twinx()
    ax2.plot(x_range, difference, label=f'Difference ({desc_1} - {desc_2})', color='tab:green', linestyle='--')
    ax2.set_ylabel('Difference', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    title_fixed_vars = ", ".join([f"{variable_descriptions.get(k, k)}={v}" for k, v in fixed_vars_dict.items()])
    plt.title(f'Trade-Off Analysis\n(Fixed: {title_fixed_vars})')
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9), bbox_transform=ax1.transAxes)

    fig.tight_layout()
    return fig

def plot_tradeoff_contour(OLS_model_1, OLS_model_2, all_alphabet_vars, x_var, y_var,
                          model_name_1, model_name_2, fixed_vars_dict, dataframe, variable_descriptions):
    """
    Generates a 2D contour plot that visualizes the trade-off between two models.
    """
    x_range = (dataframe[x_var].min(), dataframe[x_var].max())
    y_range = (dataframe[y_var].min(), dataframe[y_var].max())

    x_grid, y_grid, z_grid_1 = predict_surface(OLS_model_1, all_alphabet_vars, x_var, y_var, fixed_vars_dict, x_range, y_range)
    _, _, z_grid_2 = predict_surface(OLS_model_2, all_alphabet_vars, x_var, y_var, fixed_vars_dict, x_range, y_range)

    z_difference = z_grid_1 - z_grid_2

    desc_1 = variable_descriptions.get(model_name_1, model_name_1)
    desc_2 = variable_descriptions.get(model_name_2, model_name_2)
    x_desc = variable_descriptions.get(x_var, x_var)
    y_desc = variable_descriptions.get(y_var, y_var)

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=x_grid[0], y=y_grid[:, 0], z=z_difference,
        colorscale='RdBu', colorbar=dict(title=f'Difference<br>({desc_1} - {desc_2})'),
        name='Difference'
    ))

    fig.add_trace(go.Contour(
        x=x_grid[0], y=y_grid[:, 0], z=z_grid_1,
        contours_coloring='lines', line_color='black', name=f'Contours: {desc_1}',
        showscale=False, contours=dict(showlabels=True)
    ))

    fig.add_trace(go.Contour(
        x=x_grid[0], y=y_grid[:, 0], z=z_grid_2,
        contours_coloring='lines', line_color='grey', line_dash='dash', name=f'Contours: {desc_2}',
        showscale=False, contours=dict(showlabels=True)
    ))
    title_fixed_vars = ", ".join([f"{variable_descriptions.get(k, k)}={v}" for k, v in fixed_vars_dict.items()])
    fig.update_layout(
        title=f'2D Trade-Off Contour Plot<br>(Fixed: {title_fixed_vars})',
        xaxis_title=x_desc, yaxis_title=y_desc, height=700
    )

    return fig

def plot_combination_ranking(results_df, goal):
    """
    Generates a ranked horizontal bar chart to visualize the performance of
    different variable combinations, making it easy to identify the most impactful ones.
    """
    is_minimizing = goal in ["Minimize", "Target"]

    if is_minimizing:
        plot_df = results_df[results_df['Outcome'] >= 0].copy()
        if plot_df.empty:
            plot_df = results_df.copy()
    else:
        plot_df = results_df.copy()

    sorted_df = plot_df.sort_values('Outcome', ascending=is_minimizing).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_df) * 0.4)))

    cmap = plt.get_cmap("viridis_r" if is_minimizing else "viridis")
    colors = cmap(np.linspace(0.25, 0.85, len(sorted_df)))

    bars = ax.barh(sorted_df.index, sorted_df['Outcome'], color=colors)

    ax.set_yticks(sorted_df.index)
    ax.set_yticklabels(sorted_df['Combination'])
    ax.invert_yaxis()
    ax.set_xlabel('Best Achieved Outcome')
    ax.set_ylabel('Variable Combination')
    ax.set_title(f'Ranking of Variable Combinations (Goal: {goal})')

    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + (ax.get_xlim()[1] * 0.01)
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2., f'{width:.4f}',
                va='center', ha='left')

    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    fig.tight_layout()
    return fig

def plot_pareto_front(results_list, component1, component2, variable_descriptions):
    """
    Generates a scatter plot of the Pareto front for multi-objective optimization.
    """
    if not results_list:
        return go.Figure()

    def get_outcome(result, component):
        if component['type'] == 'model':
            return result['outcomes'][component['model_name']]
        elif component['type'] == 'factor':
            return result['factors'][component['factor_name']]

    outcomes1 = [get_outcome(r, component1) for r in results_list]
    outcomes2 = [get_outcome(r, component2) for r in results_list]
    
    name1 = component1.get('model_name') or component1.get('factor_name')
    name2 = component2.get('model_name') or component2.get('factor_name')

    desc1 = variable_descriptions.get(name1, name1)
    desc2 = variable_descriptions.get(name2, name2)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=outcomes1, y=outcomes2, mode='markers',
        name='Pareto Front Solutions', marker=dict(color='royalblue', size=10, opacity=0.8)
    ))

    fig.update_layout(
        title='<b>Pareto Front: Trade-Off Between Objectives</b>',
        xaxis_title=f'Outcome: {desc1}',
        yaxis_title=f'Outcome: {desc2}',
        legend_title_text='<b>Solutions</b>',
        height=600
    )

    return fig
    
def plot_synergy_heatmap(synergy_matrix, drug1_name, drug2_name, model_name):
    """
    Generates an interactive 2D Heatmap for synergy scores.
    Corrects "squeezed" plots by treating dose axes as categorical labels.
    """
    # Convert index (Drug 1 Doses) and columns (Drug 2 Doses) to strings.
    # This forces Plotly to space them equally (Categorical Axis) instead of numerically.
    # We round to 4 decimal places to avoid messy float labels (e.g., 0.300000004).
    y_labels = [str(round(val, 4)) if isinstance(val, (float, int)) else str(val) for val in synergy_matrix.index]
    x_labels = [str(round(val, 4)) if isinstance(val, (float, int)) else str(val) for val in synergy_matrix.columns]

    # Determine colorscale based on model
    # Gamma: < 1 is Synergy (Red), > 1 is Antagonism (Blue)
    # HSA: > 0 is Synergy (Red), < 0 is Antagonism (Blue)
    # We use 'RdBu' (Red-Blue). 
    # Usually Red=Hot/High, Blue=Cold/Low. 
    # For Gamma, Synergy is Low. So we might want to reverse it ('RdBu_r') if we want Red=Synergy.
    
    heatmap_trace = go.Heatmap(
        z=synergy_matrix.values,
        x=x_labels,
        y=y_labels,
        colorscale='RdBu_r', # Red (Low values/Synergy) to Blue (High values/Antagonism)
        colorbar=dict(title=f'{model_name} Score'),
        hovertemplate=(
            f"<b>{drug2_name} (X):</b> %{{x}}<br>"
            f"<b>{drug1_name} (Y):</b> %{{y}}<br>"
            f"<b>Score:</b> %{{z:.4f}}<extra></extra>"
        )
    )

    fig = go.Figure(data=heatmap_trace)

    fig.update_layout(
        title=f'<b>Drug Combination Synergy Matrix ({model_name})</b>',
        xaxis_title=f'Dose: {drug2_name}',
        yaxis_title=f'Dose: {drug1_name}',
        height=600,
        xaxis=dict(type='category'), # Explicitly force categorical axis
        yaxis=dict(type='category')  # Explicitly force categorical axis
    )
    return fig

def plot_pareto_frontier(model_1, model_2, independent_vars, bounds, num_samples=2000):
    """
    Returns (fig, df) so the UI can access the underlying data of the plot.
    """
    # 1. Generate random samples
    samples = []
    for _ in range(num_samples):
        sample = {}
        for i, var in enumerate(independent_vars):
            low, high = bounds[i]
            sample[var] = np.random.uniform(low, high)
        samples.append(sample)
    
    df = pd.DataFrame(samples)
    
    # 2. Predict
    pred_1 = model_1.predict(df)
    pred_2 = model_2.predict(df)
    
    df['Outcome 1'] = pred_1
    df['Outcome 2'] = pred_2
    
    # 3. Plot
    fig = px.scatter(
        df, 
        x='Outcome 1', 
        y='Outcome 2', 
        hover_data=independent_vars,
        title="Trade-off Analysis (Pareto Frontier Approximation)",
        template="plotly_white",
        opacity=0.6
    )
    
    fig.update_layout(
        xaxis_title="Model 1 Output",
        yaxis_title="Model 2 Output",
        font=dict(family="Arial", size=12),
        height=600,
        clickmode='event+select' # Enable click selection
    )
    
    return fig, df  # <--- RETURN BOTH

def plot_parallel_coordinates(model_1, model_2, independent_vars, bounds, num_samples=500):
    """
    Returns (fig, df).
    """
    samples = []
    for _ in range(num_samples):
        sample = {}
        for i, var in enumerate(independent_vars):
            low, high = bounds[i]
            sample[var] = np.random.uniform(low, high)
        samples.append(sample)
    
    df = pd.DataFrame(samples)
    pred_1 = model_1.predict(df)
    pred_2 = model_2.predict(df)
    
    df['Model 1'] = pred_1
    df['Model 2'] = pred_2
    
    dimensions = []
    for var in independent_vars:
        dimensions.append(dict(range=[df[var].min(), df[var].max()], label=var, values=df[var]))
    
    dimensions.append(dict(range=[df['Model 1'].min(), df['Model 1'].max()], label='Model 1', values=df['Model 1']))
    dimensions.append(dict(range=[df['Model 2'].min(), df['Model 2'].max()], label='Model 2', values=df['Model 2']))

    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = df['Model 1'], colorscale = 'Viridis', showscale = True,
                       cmin = df['Model 1'].min(), cmax = df['Model 1'].max(),
                       colorbar=dict(title="Model 1 Value")),
            dimensions = dimensions
        )
    )
    
    fig.update_layout(title="Parallel Coordinates", height=600)
    
    return fig, df 

def plot_synergy_heatmap(synergy_matrix, drug1_name, drug2_name, model_name):
    """
    Standard 2D Heatmap for matrix data.
    """
    fig = px.imshow(
        synergy_matrix,
        labels=dict(x=drug2_name, y=drug1_name, color="Synergy Score"),
        x=synergy_matrix.columns,
        y=synergy_matrix.index,
        color_continuous_scale="RdBu_r", 
        origin='lower'
    )
    fig.update_layout(
        title=f"Synergy Heatmap ({model_name})",
        xaxis_title=drug2_name,
        yaxis_title=drug1_name
    )
    return fig

def calculate_synergy_surface_grid(model, drug1, drug2, fixed_vars, range1, range2, method='hsa'):
    """
    Generates a grid for 3D plotting that includes:
    1. Z-axis: The Predicted Combination Effect.
    2. Color-axis: The Calculated Synergy Score (Observed - Expected).
    
    Uses the AI model to predict single-agent baselines dynamically.
    """
    # 1. Create Grid
    x_values = np.linspace(range1[0], range1[1], 50)
    y_values = np.linspace(range2[0], range2[1], 50)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    
    # Flatten for prediction
    flat_len = x_grid.size
    predict_df = pd.DataFrame(index=range(flat_len))
    predict_df[drug1] = x_grid.flatten()
    predict_df[drug2] = y_grid.flatten()
    
    # Set fixed variables
    for var, val in fixed_vars.items():
        predict_df[var] = val
        
    # --- PREDICT COMBINATION EFFECT (Z-Axis) ---
    z_combo_flat = model.predict(predict_df)
    
    # FIX: Ensure it is a numpy array (handle Series)
    if hasattr(z_combo_flat, 'values'):
        z_combo_flat = z_combo_flat.values
        
    # Clip to 0-1 (assuming inhibition)
    z_combo_flat = np.clip(z_combo_flat, 0, 1)
    
    # --- PREDICT SINGLE AGENT BASELINES (For Synergy Calc) ---
    
    # Drug 1 Alone (Drug 2 = 0)
    d1_df = predict_df.copy()
    d1_df[drug2] = 0.0
    z_d1_flat = model.predict(d1_df)
    if hasattr(z_d1_flat, 'values'): z_d1_flat = z_d1_flat.values
    z_d1_flat = np.clip(z_d1_flat, 0, 1)
    
    # Drug 2 Alone (Drug 1 = 0)
    d2_df = predict_df.copy()
    d2_df[drug1] = 0.0
    z_d2_flat = model.predict(d2_df)
    if hasattr(z_d2_flat, 'values'): z_d2_flat = z_d2_flat.values
    z_d2_flat = np.clip(z_d2_flat, 0, 1)
    
    # --- CALCULATE EXPECTATION & SYNERGY ---
    if method == 'bliss':
        # Bliss: Expected = 1 - (1-D1)*(1-D2)  (Probabilistic independence)
        expected_flat = 1 - ((1 - z_d1_flat) * (1 - z_d2_flat))
    else:
        # HSA: Expected = Max(D1, D2) (Best single agent)
        expected_flat = np.maximum(z_d1_flat, z_d2_flat)
        
    synergy_flat = z_combo_flat - expected_flat
    
    # Reshape all to grid
    z_grid = z_combo_flat.reshape(x_grid.shape)
    synergy_grid = synergy_flat.reshape(x_grid.shape)
    
    return x_grid, y_grid, z_grid, synergy_grid

def plot_3d_synergy_surface(x, y, z, synergy, d1_name, d2_name, method):
    """
    Plots the surface with Discrete Coloring (Green=Synergy, Red=Antagonism).
    """
    # Define Discrete Colorscale (Option B)
    # Mapping: -1.0 to -0.05 -> Red
    #          -0.05 to 0.05 -> Gray
    #           0.05 to 1.0  -> Green
    
    # We construct a custom scale.
    # Note: surfacecolor needs to be normalized or we use cmin/cmax
    
    fig = go.Figure(data=[go.Surface(
        x=x, 
        y=y, 
        z=z,
        surfacecolor=synergy,
        colorscale=[
            [0.0, "rgb(200, 50, 50)"],   # Deep Red (Antagonism)
            [0.45, "rgb(240, 180, 180)"], # Light Red
            [0.45, "rgb(220, 220, 220)"], # Gray start (Neutral)
            [0.55, "rgb(220, 220, 220)"], # Gray end
            [0.55, "rgb(180, 240, 180)"], # Light Green (Synergy)
            [1.0, "rgb(50, 200, 50)"]     # Deep Green
        ],
        cmin=-0.5, # Clamp range so colors don't skew
        cmax=0.5,
        colorbar=dict(title=f"Synergy ({method.upper()})")
    )])

    fig.update_layout(
        title=f"3D Synergy Surface: {d1_name} vs {d2_name}",
        scene=dict(
            xaxis_title=d1_name,
            yaxis_title=d2_name,
            zaxis_title="Inhibition (Predicted)"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig
