import pandas as pd
import numpy as np

class DrugEliminator:
    def __init__(self, drug_names, weight_toxicity=1.0, weight_efficacy=1.0):
        """
        初始化淘汰器
        :param drug_names: 候選藥物名稱列表 (例如: ['DrugA', 'DrugB', 'DrugC', 'DrugD'])
        :param weight_toxicity: 毒性在淘汰決策中的權重
        :param weight_efficacy: 療效在淘汰決策中的權重
        """
        self.drug_names = drug_names
        self.w_tox = weight_toxicity
        self.w_eff = weight_efficacy

    def _extract_coefficient(self, params_df, term):
        """從 parameters dataframe 中安全提取係數"""
        match = params_df[params_df['Term'] == term]
        if not match.empty:
            return match['Coefficient'].values[0]
        # 如果包含交互作用的順序顛倒 (如 DrugB:DrugA instead of DrugA:DrugB)
        if ':' in term:
            parts = term.split(':')
            alt_term = f"{parts[1]}:{parts[0]}"
            match_alt = params_df[params_df['Term'] == alt_term]
            if not match_alt.empty:
                return match_alt['Coefficient'].values[0]
            
            # Additional fallback check for '*' interaction representation
            alt_term2 = f"{parts[0]}*{parts[1]}"
            match_alt2 = params_df[params_df['Term'] == alt_term2]
            if not match_alt2.empty:
                return match_alt2['Coefficient'].values[0]
            
            alt_term3 = f"{parts[1]}*{parts[0]}"
            match_alt3 = params_df[params_df['Term'] == alt_term3]
            if not match_alt3.empty:
                return match_alt3['Coefficient'].values[0]

        return 0.0

    def calculate_efficacy(self, params_df, drug):
        """計算單一模型中的藥物療效分數 (越正越好)"""
        # 1. 單一藥物主效應 (越負代表細胞存活率降越多 -> 療效越好)
        linear_effect = self._extract_coefficient(params_df, drug)
        
        # 2. 交互作用效應
        interaction_effect = 0.0
        for other_drug in self.drug_names:
            if other_drug != drug:
                # OLS formula API output commonly uses : for interactions
                term = f"{drug}:{other_drug}"
                interaction_effect += self._extract_coefficient(params_df, term)
                
        # 效應反轉：負的係數代表存活率下降（好），所以加負號讓正值代表好
        return -linear_effect - interaction_effect

    def calculate_toxicity(self, params_df, drug):
        """計算單一模型中的藥物毒性分數 (越正代表毒性越強)"""
        # 對正常細胞，主效應越負代表存活率越低 -> 毒性越強
        linear_effect = self._extract_coefficient(params_df, drug)
        return -linear_effect

    def evaluate(self, normal_model_df, cancer_models_dfs):
        """
        :param normal_model_df: 正常細胞的迴歸係數 DataFrame
        :param cancer_models_dfs: 癌細胞的迴歸係數 DataFrame 列表 (支援單一或多種癌細胞)
        :return: 排序後的淘汰建議 DataFrame
        """
        results = []
        
        for drug in self.drug_names:
            # 1. 計算毒性 (Toxicity)
            tox_score = self.calculate_toxicity(normal_model_df, drug)
            
            # 2. 計算平均療效 (Average Efficacy 支援多種癌細胞)
            eff_scores = [self.calculate_efficacy(df, drug) for df in cancer_models_dfs]
            avg_eff_score = np.mean(eff_scores) if eff_scores else 0.0
            
            # 3. 計算淘汰優先權 (Priority = Tox - Eff)
            # 毒性越高越該淘汰(+)，療效越好越不該淘汰(-)
            elimination_priority = (self.w_tox * tox_score) - (self.w_eff * avg_eff_score)
            
            results.append({
                'Drug': drug,
                'Toxicity_Score (Normal)': round(tox_score, 4),
                'Avg_Efficacy_Score (Cancer)': round(avg_eff_score, 4),
                'Elimination_Priority': round(elimination_priority, 4)
            })
            
        # 轉成 DataFrame 並由最該淘汰 (分數最高) 排到最不該淘汰 (分數最低)
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Elimination_Priority', ascending=False).reset_index(drop=True)
        
        return results_df

    @staticmethod
    def get_formula_latex():
        """Returns the formula used for calculation in LaTeX format."""
        return r'''
        \begin{align*}
        E_k &= -\beta_k - \sum_{j \neq k} \beta_{kj} \quad \text{(Efficacy)} \\
        T_k &= -\beta_k \quad \text{(Toxicity)} \\
        P_k &= W_{tox} \times T_k - W_{eff} \times \bar{E}_k \quad \text{(Elimination Priority)}
        \end{align*}
        '''
