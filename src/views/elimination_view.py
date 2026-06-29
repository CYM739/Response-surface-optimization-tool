import streamlit as st
import pandas as pd
from logic.models import OLSWrapper
from logic.drug_elimination import DrugEliminator

def render():
    st.header("🗑️ Drug Elimination Score")
    st.markdown("Evaluate which drug to eliminate based on toxicity to normal cells and lack of efficacy on cancer cells. The formula calculates elimination priority as follows:")
    st.latex(DrugEliminator.get_formula_latex())

    # Filter available models for OLS wrappers because we need coefficients
    if not st.session_state.get('wrapped_models'):
        st.warning("No models found. Please load and analyze data first.")
        return

    ols_models = {
        name: model for name, model in st.session_state.wrapped_models.items() 
        if isinstance(model, OLSWrapper)
    }

    if not ols_models:
        st.warning("Only OLS models can be used for this scoring system, since it relies on linear coefficients. No OLS models were found.")
        return

    model_names = list(ols_models.keys())

    # Need at least two models to perform meaningful tradeoff analysis
    if len(model_names) < 2:
        st.warning("This requires at least two OLS models (one for normal cells, one for cancer cells). Make sure your analysis generates multiple models.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Selection")
        normal_model_sel = st.selectbox("Select Normal Cell Model (Toxicity)", model_names, help="The model representing normal cell survival rate. You want to avoid decreasing this.")
        
        # Determine remaining choices for cancer models
        cancer_model_choices = [m for m in model_names if m != normal_model_sel]
        cancer_models_sel = st.multiselect("Select Cancer Cell Model(s) (Efficacy)", cancer_model_choices, default=cancer_model_choices[:1] if cancer_model_choices else None, help="The model(s) representing cancer cell survival rate. You want to decrease this.")

    with col2:
        st.subheader("Weight Configuration")
        weight_tox = st.slider("Weight: Toxicity", min_value=0.0, max_value=5.0, value=1.0, step=0.1, help="Higher values penalize drugs that kill normal cells.")
        weight_eff = st.slider("Weight: Efficacy", min_value=0.0, max_value=5.0, value=1.0, step=0.1, help="Higher values reward drugs that successfully kill cancer cells independently or synergistically.")
        
    st.divider()
    st.markdown("### Underlying OLS Equations")
    st.markdown("The $\\beta$ values in the top formula refer to the coefficients in these linear equations:")
    st.info(f"**Normal Model Equation:**\n`{ols_models[normal_model_sel].formula}`")
    for c_model in cancer_models_sel if cancer_models_sel else []:
        st.info(f"**Cancer Model ({c_model}) Equation:**\n`{ols_models[c_model].formula}`")
        
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Elimination Analysis", type="primary"):
        if not cancer_models_sel:
            st.error("Please select at least one cancer cell model.")
            return

        # Fetch the underlying params DataFrame from models
        normal_model_df = ols_models[normal_model_sel].get_params_df()
        cancer_models_dfs = [ols_models[m].get_params_df() for m in cancer_models_sel]

        # Get drug names from independent variables
        # We assume independent variables across models are mostly identical.
        # We'll grab it from the normal model.
        drug_names = ols_models[normal_model_sel].independent_vars

        if not drug_names:
            st.error("Could not extract independent variable/drug names from the normal cell model.")
            return

        try:
            eliminator = DrugEliminator(
                drug_names=drug_names,
                weight_toxicity=weight_tox,
                weight_efficacy=weight_eff
            )

            results_df = eliminator.evaluate(normal_model_df, cancer_models_dfs)

            st.success("Priorities calculated successfully!")
            
            # Display results
            st.subheader("Elimination Rankings")
            st.markdown("Higher **Elimination Priority** implies the drug is toxic and has low efficacy. **Top drugs should be eliminated.**")
            st.dataframe(
                results_df.style.background_gradient(cmap="Reds", subset=['Elimination_Priority']),
                width="stretch"
            )
            
            # Provide an expander for debugging params
            with st.expander("View Raw Model Coefficients (Debug)"):
                st.write(f"**Normal Model ({normal_model_sel}) Coefficients**")
                st.dataframe(normal_model_df)
                for c_name, c_df in zip(cancer_models_sel, cancer_models_dfs):
                    st.write(f"**Cancer Model ({c_name}) Coefficients**")
                    st.dataframe(c_df)
                    
        except Exception as e:
            st.error(f"Error during calculations: {e}")
