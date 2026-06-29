# src/views_edu/dashboard_view.py
import streamlit as st
import pandas as pd
import altair as alt
from logic.data_processing import analyze_csv, run_analysis, expand_terms
from logic.optimization import run_classic_multi_objective_optimization, run_optimization, objective_function
from utils.state_management import reset_state, clear_optimizer_results
from utils.ui_helpers import format_variable_options, display_surface_plot
from views.library_view import load_library, save_library
from io import StringIO

def render():
    """渲染完整的互動式儀表板介面。"""
    st.title("🔬 AIPRS 互動儀表板 (教育版)")

    with st.sidebar:
        render_sidebar()

    st.header("分析結果儀表板")
    if not st.session_state.get('analysis_done'):
        st.info("👈 請從左側的控制面板開始，建立或載入專案並執行分析。")
        return

    render_main_dashboard()


def render_sidebar():
    """渲染側邊欄的所有控制元件。"""
    st.header("控制面板")
    
    with st.expander("📂 專案管理", expanded=True):
        manage_projects()

    if st.session_state.get('analysis_done'):
        st.divider()
        with st.expander("🎯 單目標優化器", expanded=False):
            run_single_objective_optimizer()
        
        st.divider()
        with st.expander("⚖️ 多目標優化器", expanded=True):
            run_multi_objective_optimizer()


def manage_projects():
    """側邊欄中的專案建立、載入、刪除功能。"""
    st.subheader("建立新專案")
    uploaded_file = st.file_uploader("上傳 CSV 檔案", type=["csv"])
    project_name_input = st.text_input("專案名稱")
    if st.button("建立並載入專案"):
        if uploaded_file and project_name_input:
            library = load_library()
            if project_name_input in library:
                st.error("專案名稱已存在！")
            else:
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='ms950')
                
                df_json = df.to_json(orient='split')
                library[project_name_input] = {"data_df_json": df_json, "analysis_runs": {}}
                save_library(library)
                st.success(f"專案 '{project_name_input}' 已建立！")
                reset_state()
                (st.session_state.exp_df, st.session_state.all_vars, st.session_state.independent_vars,
                 st.session_state.dependent_vars, st.session_state.variable_stats,
                 _, st.session_state.unique_variable_values,
                 st.session_state.variable_descriptions,
                 st.session_state.detected_binary_vars) = analyze_csv(df)
                st.session_state.processed_file = project_name_input
                st.rerun()

    st.divider()
    
    library = load_library()
    if library:
        st.subheader("管理現有專案")
        selected_project = st.selectbox("選擇專案", sorted(list(library.keys())))
        
        col1, col2 = st.columns(2)
        if col1.button("載入專案", width="stretch"):
            project_data = library[selected_project]
            df = pd.read_json(StringIO(project_data['data_df_json']), orient='split')
            reset_state()
            (st.session_state.exp_df, st.session_state.all_vars, st.session_state.independent_vars,
             st.session_state.dependent_vars, st.session_state.variable_stats,
             _, st.session_state.unique_variable_values,
             st.session_state.variable_descriptions,
             st.session_state.detected_binary_vars) = analyze_csv(df)
            st.session_state.processed_file = selected_project
            st.success(f"已載入專案 '{selected_project}'")
            st.rerun()

        if col2.button("刪除專案", type="secondary", width="stretch"):
            del library[selected_project]
            save_library(library)
            if st.session_state.get('processed_file') == selected_project:
                reset_state()
            st.success(f"已刪除專案 '{selected_project}'")
            st.rerun()

    if st.session_state.get('processed_file') and not st.session_state.get('analysis_done'):
        st.divider()
        st.subheader(f"為「{st.session_state.processed_file}」執行分析")
        if st.button("執行 OLS 模型分析", type="primary", width="stretch"):
            with st.spinner("正在建立模型..."):
                df_for_analysis = st.session_state.exp_df.copy()
                expand_terms(df_for_analysis, st.session_state.independent_vars)
                st.session_state.expanded_df = df_for_analysis
                
                current_wrapped_models = {}
                for dep_var in st.session_state.dependent_vars:
                    wrapped_model = run_analysis(
                        df_for_analysis, st.session_state.independent_vars, 
                        dep_var, 'Polynomial OLS', {}
                    )
                    current_wrapped_models[dep_var] = wrapped_model

                st.session_state.wrapped_models = current_wrapped_models
                st.session_state.analysis_done = True
                st.session_state.active_analysis_run = "預設 OLS 分析"
                st.success("模型分析完成！")
                st.rerun()

def run_single_objective_optimizer():
    """側邊欄中的單目標優化器介面與邏輯。"""
    model_name = list(st.session_state.wrapped_models.keys())[0]
    st.info(f"目前的優化目標為：**{st.session_state.variable_descriptions.get(model_name, model_name)}**")
    
    target_value = st.number_input("請輸入目標值", value=0.0, format="%.4f", key="single_opt_target", on_change=clear_optimizer_results)
    fallback_goal = st.radio(
        "若無法精確達成，則尋找最接近的：",
        ("大於目標的值", "小於目標的值"),
        horizontal=True, key="single_opt_fallback", on_change=clear_optimizer_results
    )

    bounds = []
    for var in st.session_state.independent_vars:
        min_v, second_min_v, max_v = st.session_state.variable_stats[var]
        desc = st.session_state.variable_descriptions.get(var, var)
        bounds.append(st.slider(
            f"變數「{desc}」的範圍", 
            float(min_v), float(max_v), (float(second_min_v), float(max_v)),
            key=f"opt_bound_{var}"
        ))

    if st.button("執行單目標優化", type="primary", width="stretch"):
        selected_model = st.session_state.wrapped_models[model_name]
        start_points = [b[0] for b in bounds]
        
        with st.spinner("正在尋找最佳解..."):
            obj_fun_exact = lambda x: abs(objective_function(x, selected_model, st.session_state.independent_vars) - target_value)
            result = run_optimization(obj_fun_exact, bounds, start_points, [], 'SLSQP (Local)', {})

            if not result.success or result.fun > 1e-4:
                st.warning("無法找到精確匹配，正在尋找替代方案...")
                fallback_constraints = []
                if fallback_goal == "大於目標的值":
                    obj_fun_fallback = lambda x: objective_function(x, selected_model, st.session_state.independent_vars)
                    cons = {'type': 'ineq', 'fun': lambda x: objective_function(x, selected_model, st.session_state.independent_vars) - target_value}
                    fallback_constraints.append(cons)
                else:
                    obj_fun_fallback = lambda x: -objective_function(x, selected_model, st.session_state.independent_vars)
                    cons = {'type': 'ineq', 'fun': lambda x: target_value - objective_function(x, selected_model, st.session_state.independent_vars)}
                    fallback_constraints.append(cons)
                result = run_optimization(obj_fun_fallback, bounds, start_points, fallback_constraints, 'SLSQP (Local)', {})
        
        if result.success:
            st.success("優化成功！")
            final_outcome = objective_function(result.x, selected_model, st.session_state.independent_vars)
            results_dict = dict(zip(st.session_state.independent_vars, result.x))
            st.session_state.single_opt_results = {
                "dosages": results_dict,
                "outcome": final_outcome,
                "model_name": model_name
            }
            
            # [FIX START] Force update the slider values for the dashboard
            for var, val in results_dict.items():
                st.session_state[f"main_plot_fixed_{var}"] = float(val)
            # [FIX END]
            
            st.rerun()
        else:
            st.error(f"優化失敗: {result.message}")
            st.session_state.single_opt_results = None

def run_multi_objective_optimizer():
    """側邊欄中的多目標優化器介面與邏輯。"""
    st.info("設定一個主要目標和一個次要目標，找到兼顧兩者的最佳解。")
    
    formatted_models = format_variable_options(st.session_state.wrapped_models.keys())
    
    if len(formatted_models) < 2:
        st.warning("您需要至少兩個結果模型才能使用多目標優化器。")
        return

    st.markdown("**目標一 (限制條件)**")
    model_1_formatted = st.selectbox("選擇要限制的結果模型", options=formatted_models, key="edu_multi_model_1", on_change=clear_optimizer_results)
    model_1_name = model_1_formatted.split(":")[0]

    r_min = st.number_input(f"可接受的最小值", value=0.0, format="%.4f", key="edu_multi_min")
    r_max = st.number_input(f"可接受的最大值", value=0.0, format="%.4f", key="edu_multi_max_v2")

    st.markdown("**目標二 (主要優化目標)**")
    model_2_options = [m for m in formatted_models if not m.startswith(model_1_name)]
    model_2_formatted = st.selectbox("選擇要優化的結果模型", options=model_2_options, key="edu_multi_model_2", on_change=clear_optimizer_results)
    model_2_name = model_2_formatted.split(":")[0]

    target_model_goal = st.radio("優化目標", ("最大化", "最小化"), key="edu_multi_goal", horizontal=True)

    st.markdown("**變數搜尋範圍**")
    bounds = []
    for var in st.session_state.independent_vars:
        min_v, second_min_v, max_v = st.session_state.variable_stats[var]
        desc = st.session_state.variable_descriptions.get(var, var)
        bounds.append(st.slider(
            f"變數「{desc}」的範圍",
            float(min_v), float(max_v), (float(second_min_v), float(max_v)),
            key=f"opt_bound_multi_{var}"
        ))

    if st.button("執行多目標優化", type="primary", width="stretch"):
        model_1 = st.session_state.wrapped_models[model_1_name]
        model_2 = st.session_state.wrapped_models[model_2_name]
        start_points = [b[0] for b in bounds]

        with st.spinner("正在執行多目標優化..."):
            final_dosages, status_message = run_classic_multi_objective_optimization(
                model_1, model_2, st.session_state.independent_vars, bounds, start_points,
                r_min, r_max, "Maximize" if target_model_goal == "最大化" else "Minimize", 'SLSQP (Local)', {}
            )

        if final_dosages is not None:
            st.success("多目標優化成功！")
            outcome_1 = objective_function(final_dosages, model_1, st.session_state.independent_vars)
            outcome_2 = objective_function(final_dosages, model_2, st.session_state.independent_vars)
            
            results_dict = dict(zip(st.session_state.independent_vars, final_dosages))
            st.session_state.multi_opt_results = {
                "dosages": results_dict,
                "outcome_1": outcome_1,
                "outcome_2": outcome_2,
                "model_1_name": model_1_name,
                "model_2_name": model_2_name,
            }
            
            # [FIX START] Force update the slider values for the dashboard
            for var, val in results_dict.items():
                st.session_state[f"main_plot_fixed_{var}"] = float(val)
            # [FIX END]

            st.rerun()
        else:
            st.error(f"優化失敗: {status_message}")

def render_main_dashboard():
    """渲染主畫面的儀表板內容。"""
    
    opt_results = st.session_state.get("single_opt_results")
    multi_opt_results = st.session_state.get("multi_opt_results")
    
    point_to_plot = None
    if multi_opt_results:
        point_to_plot = multi_opt_results.get('dosages')
    elif opt_results:
        point_to_plot = opt_results.get('dosages')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 3D 反應曲面圖")
        
        model_to_plot_1 = list(st.session_state.wrapped_models.keys())[0]
        if multi_opt_results:
            model_choices = [multi_opt_results['model_1_name'], multi_opt_results['model_2_name']]
            formatted_model_choices = format_variable_options(model_choices)
            selected_model_formatted = st.selectbox("選擇主要顯示的模型", formatted_model_choices, key="main_plot_model_select")
            model_to_plot_1 = selected_model_formatted.split(":")[0]

        c1, c2 = st.columns(2)
        formatted_vars = format_variable_options(st.session_state.independent_vars)
        x_var = c1.selectbox("X 軸變數", formatted_vars, key="main_plot_x").split(":")[0]
        y_options = [v for v in formatted_vars if not v.startswith(x_var)]
        y_var = c2.selectbox("Y 軸變數", y_options, key="main_plot_y").split(":")[0]

        fixed_vars = {}
        other_vars = [v for v in st.session_state.independent_vars if v not in [x_var, y_var]]
        if other_vars:
            for var in other_vars:
                min_v, _, max_v = st.session_state.variable_stats[var]
                desc = st.session_state.variable_descriptions.get(var, var)
                
                # Logic: If point_to_plot exists, we prefer that. 
                # BUT st.slider relies on session state key if it exists.
                # The FIX above ensures session state key is updated.
                default_val = point_to_plot[var] if point_to_plot and var in point_to_plot else float((min_v+max_v)/2)
                
                fixed_vars[var] = st.slider(f"固定值: {desc}", float(min_v), float(max_v), default_val, key=f"main_plot_fixed_{var}")

        model_to_plot_2 = None
        optimized_point_2 = None
        if multi_opt_results:
            with st.expander("圖表顯示設定 (Overlay)"):
                overlay_second_model = st.checkbox("疊加第二個模型曲面", value=True)
                if overlay_second_model:
                    if model_to_plot_1 == multi_opt_results['model_1_name']:
                        model_to_plot_2 = multi_opt_results['model_2_name']
                    else:
                        model_to_plot_2 = multi_opt_results['model_1_name']
                    optimized_point_2 = point_to_plot

        # --- Chart Appearance Controls ---
        colorscale_options = ['Viridis', 'Plasma', 'Jet', 'Cividis', 'Hot', 'Cool', 'Blues', 'Greens', 'RdBu', 'Greys']
        
        with st.expander("🎨 圖表外觀設定 (Chart Appearance)"):
            st.markdown("##### 標題與字體 (Titles & Fonts)")
            t1, t2 = st.columns(2)
            custom_main_title = t1.text_input("圖表主標題 (Main Title)", placeholder="預設標題")
            axis_title_font_size = t2.slider("軸標題字體大小 (Axis Title Size)", 8, 24, 12)
            
            t3, t4, t5 = st.columns(3)
            custom_x_title = t3.text_input("X 軸標題", placeholder="預設")
            custom_y_title = t4.text_input("Y 軸標題", placeholder="預設")
            custom_z_title = t5.text_input("Z 軸標題", placeholder="預設")
            axis_tick_font_size = st.slider("刻度數字字體大小 (Tick Font Size)", 8, 24, 10)

            st.markdown("##### 顏色與格線 (Colors & Grids)")
            c1, c2, c3, c4 = st.columns(4)
            colorscale = c1.selectbox("顏色主題 (Color Scale)", colorscale_options, index=1)
            show_x_grid = c2.checkbox("顯示 X 軸網格", value=True)
            show_y_grid = c3.checkbox("顯示 Y 軸網格", value=True)
            show_surface_grid = c4.toggle("顯示曲面網格 (Show Surface Grid)", value=True)
            show_actual_data = st.toggle("顯示實際數據點 (Show Actual Data)", value=True)

            st.markdown("##### Z 軸範圍 (Z-Axis Range)")
            enable_z_limit = st.checkbox("手動設定 Z 軸範圍 (Manual Range)", value=False)
            z_range = None
            if enable_z_limit:
                z_c1, z_c2 = st.columns(2)
                z_min = z_c1.number_input("最小 Z 值 (Min Z)", value=0.0)
                z_max = z_c2.number_input("最大 Z 值 (Max Z)", value=100.0)
                z_range = [z_min, z_max]
            
            st.markdown("##### 匯出設定 (Export)")
            download_scale = st.number_input("圖片下載解析度倍率 (Download Scale)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

        plot_params = {
            'x_var': x_var,
            'y_var': y_var,
            'z_var_1': model_to_plot_1,
            'fixed_vars_dict_1': fixed_vars,
            'variable_descriptions': st.session_state.variable_descriptions,
            'show_actual_data': show_actual_data,
            'optimized_point': point_to_plot,
            'z_var_2': model_to_plot_2,
            'fixed_vars_dict_2': fixed_vars if model_to_plot_2 else None,
            'optimized_point_2': optimized_point_2,
            
            # Appearance Settings
            'main_title': custom_main_title or None,
            'x_title': custom_x_title or None,
            'y_title': custom_y_title or None,
            'z_title': custom_z_title or None,
            'axis_title_font_size': axis_title_font_size,
            'axis_tick_font_size': axis_tick_font_size,
            'colorscale_1': colorscale,
            'show_x_grid': show_x_grid,
            'show_y_grid': show_y_grid,
            'show_surface_grid': show_surface_grid,
            'z_range': z_range,
        }
        
        # Configuration for the plotly figure (passed separately)
        plot_config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{model_to_plot_1}_surface_plot',
                'height': 700,
                'width': 700,
                'scale': download_scale
            }
        }
        
        display_surface_plot(plot_params, plot_config)

    with col2:
        st.subheader("🎯 優化結果")
        if not opt_results and not multi_opt_results:
            st.info("請在左側的優化器中執行一次優化，結果將會顯示於此。")

        if multi_opt_results:
            with st.container(border=True):
                st.markdown("##### ⚖️ 多目標優化結果")
                m1_name = multi_opt_results['model_1_name']
                m2_name = multi_opt_results['model_2_name']
                m1_desc = st.session_state.variable_descriptions.get(m1_name, m1_name)
                m2_desc = st.session_state.variable_descriptions.get(m2_name, m2_name)

                st.metric(label=f"結果一 ({m1_desc})", value=f"{multi_opt_results['outcome_1']:.4f}")
                st.metric(label=f"結果二 ({m2_desc})", value=f"{multi_opt_results['outcome_2']:.4f}")
                st.markdown("**最佳平衡點的變因組合:**")
                dosages = multi_opt_results['dosages']
                descriptive_dosages = {st.session_state.variable_descriptions.get(k, k): v for k, v in dosages.items()}
                df_dosages = pd.DataFrame(descriptive_dosages.items(), columns=['變因', '最佳值'])
                st.dataframe(df_dosages)


        if opt_results:
            with st.container(border=True):
                st.markdown("##### 🎯 單目標優化結果")
                model_name = opt_results['model_name']
                st.metric(
                    label=f"最佳結果 ({st.session_state.variable_descriptions.get(model_name, model_name)})",
                    value=f"{opt_results['outcome']:.4f}"
                )
                st.markdown("**最佳變因組合:**")
                dosages = opt_results['dosages']
                descriptive_dosages = {st.session_state.variable_descriptions.get(k, k): v for k, v in dosages.items()}
                df_dosages = pd.DataFrame(descriptive_dosages.items(), columns=['變因', '最佳值'])
                st.dataframe(df_dosages)

    # --- 簡化後的模型摘要區塊 ---
    st.subheader("📊 模型係數影響力")
    model_to_show_summary = model_to_plot_1
    wrapped_model = st.session_state.wrapped_models[model_to_show_summary]
    st.markdown(f"**正在顯示模型「{st.session_state.variable_descriptions.get(model_to_show_summary, model_to_show_summary)}」的係數影響力**")

    if hasattr(wrapped_model, 'get_params_df'):
        params_df = wrapped_model.get_params_df()
        
        # Exclude intercept for better visualization scale
        params_to_plot = params_df[params_df['Term'] != 'Intercept']
        if not params_to_plot.empty:
            chart = alt.Chart(params_to_plot).mark_bar().encode(
                x=alt.X('Coefficient', title='係數大小'),
                y=alt.Y('Term', title='模型項目', sort='-x')
            ).properties(
                title='模型係數影響力視覺化'
            )
            st.altair_chart(chart, width="stretch")
        else:
            st.info("模型中沒有可供視覺化的係數。")
    else:
         # Fallback for other model types that don't have get_params_df
         summary_text = wrapped_model.get_summary()
         st.code(summary_text)
