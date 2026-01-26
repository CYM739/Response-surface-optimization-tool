# src/views/synergy_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from logic.data_processing import calculate_high_throughput_synergy
from logic.plotting import calculate_synergy_surface_grid, plot_3d_synergy_surface
from utils.ui_helpers import format_variable_options

def render():
    """Renders the Row-Based High-Throughput Synergy Analysis Tab."""
    st.subheader("🧪 High-Throughput Synergy Screening")
    
    if 'analysis_done' not in st.session_state or not st.session_state.analysis_done:
        st.info("Please load a project and run an analysis first (to enable AI-assisted reference predictions).")
        return

    st.markdown("""
    This tool analyzes drug combinations row-by-row. It supports **3+ drug combinations** and **sparse datasets**.
    If specific single-agent reference doses are missing, it uses your trained AI model to estimate them.
    """)

    with st.container(border=True):
        st.markdown("##### ⚙️ Screening Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Multi-select for drugs (N-way combinations)
            drug_options = st.session_state.independent_vars
            selected_drugs = st.multiselect(
                "Select All Drug Columns (Dose)", 
                options=drug_options,
                default=drug_options[:2] if len(drug_options) >=2 else drug_options
            )
            
        with col2:
            # Select Effect
            effect_options = st.session_state.dependent_vars
            selected_effect = st.selectbox("Select Effect/Response Column", options=effect_options)
            
            # Select Model for Imputation
            model_names = list(st.session_state.get('wrapped_models', {}).keys())
            if model_names:
                selected_model_name = st.selectbox("Select AI Model (for filling missing controls)", options=model_names)
            else:
                st.warning("No models found. Analysis will fail if reference data is missing.")
                selected_model_name = None

    if st.button("🚀 Run Screening Analysis", type="primary"):
        if not selected_drugs or not selected_effect:
            st.error("Please select drugs and an effect column.")
            return

        with st.spinner("Analyzing combinations... detecting direction... predicting missing controls..."):
            try:
                # Get the model object
                model_obj = st.session_state.wrapped_models.get(selected_model_name) if selected_model_name else None
                
                # Run Logic
                df_results, direction = calculate_high_throughput_synergy(
                    df=st.session_state.exp_df,
                    dose_cols=selected_drugs,
                    effect_col=selected_effect,
                    model=model_obj
                )
                
                # Store in session state
                st.session_state.synergy_results = df_results
                st.session_state.synergy_direction = direction
                st.session_state.synergy_active_drugs = selected_drugs
                st.session_state.synergy_active_model = selected_model_name # Save for plotting
                
                st.success(f"Analysis Complete! Auto-detected Data Type: **{direction.title()}**")
                
            except Exception as e:
                st.error(f"Analysis Failed: {e}")
                st.exception(e)

    # --- RESULTS DISPLAY ---
    if st.session_state.get('synergy_results') is not None:
        results = st.session_state.synergy_results
        direction = st.session_state.get('synergy_direction', 'Unknown')
        
        st.write("---")
        st.subheader("📊 Analysis Results")
        
        # 1. Summary Metrics
        synergy_hits = results[results['HSA_Synergy'] > 0.1]
        antagonism_hits = results[results['HSA_Synergy'] < -0.1]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Combinations", len(results[results['Analysis_Note'] != 'Single Agent / Control']))
        m2.metric("Synergistic Hits (HSA > 0.1)", len(synergy_hits))
        m3.metric("Antagonistic Hits (HSA < -0.1)", len(antagonism_hits))
        
        # 2. Main Data Table
        st.markdown("#### Detailed Data Table")
        
        # Filter options
        filter_col, _ = st.columns([1,2])
        show_only_combos = filter_col.checkbox("Hide Single Agents / Controls", value=True)
        
        display_df = results.copy()
        if show_only_combos:
            display_df = display_df[display_df['Analysis_Note'] != 'Single Agent / Control']
        
        # Color highlighting
        def color_synergy(val):
            if pd.isna(val): return ''
            if val > 0.1: return 'background-color: #d4edda; color: #155724' # Green
            if val < -0.1: return 'background-color: #f8d7da; color: #721c24' # Red
            return ''
            
        st.dataframe(
            display_df.style.map(color_synergy, subset=['HSA_Synergy', 'Bliss_Synergy'])
                     .format("{:.4f}", subset=['HSA_Synergy', 'Bliss_Synergy', 'std_effect']),
            use_container_width=True,
            height=400
        )
        
        # Download
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results CSV", csv, "synergy_screening_results.csv", "text/csv")
        
        # 3. Visualization Tabs
        st.write("---")
        st.subheader("📈 Visualization")
        
        tab1, tab2, tab3 = st.tabs(["Distribution", "Scatter Plot", "🧊 3D Synergy Surface"])
        
        with tab1:
            # Histogram of scores
            fig_hist = px.histogram(
                display_df, 
                x="HSA_Synergy", 
                nbins=30, 
                title="Distribution of Synergy Scores (HSA)",
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with tab2:
            # Scatter: Total Dose vs Synergy
            display_df['Total_Dose'] = display_df[st.session_state.synergy_active_drugs].sum(axis=1)
            fig_scat = px.scatter(
                display_df,
                x="Total_Dose",
                y="HSA_Synergy",
                color="Analysis_Note",
                hover_data=st.session_state.synergy_active_drugs,
                title="Total Dose vs. Synergy Score"
            )
            fig_scat.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_scat, use_container_width=True)

        with tab3:
            # 3D Surface with Synergy Overlay
            st.markdown("**Visualize Interaction Surface:** Shape = Effect, Color = Synergy")
            
            # Use active drugs from analysis
            active_drugs = st.session_state.synergy_active_drugs
            if len(active_drugs) < 2:
                st.warning("Need at least 2 drugs for a surface plot.")
            else:
                c1, c2, c3 = st.columns(3)
                d1 = c1.selectbox("X-Axis Drug", active_drugs, index=0)
                d2 = c2.selectbox("Y-Axis Drug", active_drugs, index=1 if len(active_drugs)>1 else 0)
                method = c3.radio("Synergy Method", ["HSA", "Bliss"], index=0)
                
                # Get Model
                model_name = st.session_state.get('synergy_active_model')
                model_obj = st.session_state.wrapped_models.get(model_name)
                
                if not model_obj:
                    st.error("Model not found. Please re-run analysis.")
                else:
                    if st.button("Draw Surface"):
                        # Define ranges (Auto-detect from data)
                        max_d1 = st.session_state.exp_df[d1].max()
                        max_d2 = st.session_state.exp_df[d2].max()
                        
                        # Prepare fixed vars (others set to 0)
                        fixed_vars = {d: 0.0 for d in st.session_state.independent_vars if d not in [d1, d2]}
                        
                        with st.spinner("Calculating synergy surface..."):
                            x_g, y_g, z_g, syn_g = calculate_synergy_surface_grid(
                                model=model_obj,
                                drug1=d1, drug2=d2,
                                fixed_vars=fixed_vars,
                                range1=(0, max_d1),
                                range2=(0, max_d2),
                                method=method.lower()
                            )
                            
                            fig_surf = plot_3d_synergy_surface(
                                x_g, y_g, z_g, syn_g, d1, d2, method
                            )
                            st.plotly_chart(fig_surf, use_container_width=True)
