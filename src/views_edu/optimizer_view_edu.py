# src/views_edu/optimizer_view_edu.py
import streamlit as st
import pandas as pd
from logic.optimization import run_bayesian_optimization
from utils.ui_helpers import format_variable_options, validate_bounds_for_ai
from utils.state_management import clear_optimizer_results

def render():
    """渲染 AI 智慧優化介面 (教育版) 的所有 UI 元件與邏輯。"""
    st.header("AI 智慧優化")
    st.info(
        """
        這個工具使用先進的貝氏優化演算法，能高效率地幫您在複雜的多維度空間中，
        找到能達成「最大化」、「最小化」或「特定目標值」的最佳變因組合。
        """
    )
    
    formatted_models = format_variable_options(st.session_state.wrapped_models.keys())

    # --- 優化目標設定 ---
    with st.container(border=True):
        st.subheader("第一步：設定優化目標")
        col1, col2 = st.columns(2)

        selected_model_formatted = col1.selectbox(
            "選擇要優化的結果模型",
            options=formatted_models,
            key="edu_bopt_model",
            on_change=clear_optimizer_results
        )
        model_to_optimize = selected_model_formatted.split(":")[0]

        goal = col2.radio(
            "您的優化目標是？",
            ("最大化 (Maximize)", "最小化 (Minimize)"),
            key="edu_bopt_goal",
            horizontal=True,
            on_change=clear_optimizer_results
        )
        goal_key = goal.split(" ")[0]

    # --- 變數範圍設定 ---
    with st.container(border=True):
        st.subheader("第二步：設定變數的搜尋範圍 (邊界)")
        st.markdown("請為每一個變因設定一個合理的搜尋範圍，AI 將會在此範圍內尋找最佳解。")
        bounds = []
        for var in st.session_state.independent_vars:
            min_val, _, max_val = st.session_state.variable_stats[var]
            desc = st.session_state.variable_descriptions.get(var, var)
            
            c1, c2 = st.columns(2)
            min_bound = c1.number_input(f"變數「{desc}」的最小值", value=float(min_val), key=f"edu_bopt_min_{var}")
            max_bound = c2.number_input(f"變數「{desc}」的最大值", value=float(max_val), key=f"edu_bopt_max_{var}")
            bounds.append((min_bound, max_bound))
    
    # --- 執行與結果 ---
    if st.button("🤖 啟動 AI 優化", type="primary", width="stretch"):
        clear_optimizer_results()
        if not validate_bounds_for_ai(bounds, st.session_state.independent_vars, st.session_state.variable_descriptions):
            st.stop()
        
        with st.spinner("AI 正在進行智慧搜尋... 這可能需要一點時間。"):
            try:
                selected_model = st.session_state.wrapped_models[model_to_optimize]
                results = run_bayesian_optimization(
                    OLS_model=selected_model,
                    all_alphabet_vars=st.session_state.independent_vars,
                    bounds=bounds,
                    goal=goal_key,
                    n_calls=25, # 固定迭代次數
                    n_initial_points=10, # 固定初始點
                    variable_descriptions=st.session_state.variable_descriptions,
                )
                st.session_state.bayesian_opt_results = results
                st.success("AI 優化完成！")
            except Exception as e:
                st.error(f"優化過程中發生錯誤: {e}")

    # --- 顯示結果 ---
    if st.session_state.get("bayesian_opt_results"):
        st.divider()
        st.subheader("優化結果")
        res = st.session_state.bayesian_opt_results
        
        st.metric(
            label=f"找到的最佳結果 ({goal})",
            value=f"{res['outcome']:.4f}"
        )

        st.markdown("**達成此結果的最佳變因組合：**")
        descriptive_vars = [st.session_state.variable_descriptions.get(var, var) for var in st.session_state.independent_vars]
        dosages_df = pd.DataFrame({'變因': descriptive_vars, '最佳值': res['dosages']})
        st.dataframe(dosages_df, width="stretch")

        st.markdown("**收斂過程圖：**")
        st.pyplot(res['convergence_plot'])