import os, sys, time, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from get_train_test_data import get_train_test_data
from sklearn.model_selection import train_test_split
import joblib
from xgboost import XGBClassifier


class Explainer:
    def __init__(self):
        self.add_project_folder_to_pythonpath()

        f = open(os.path.join("data", "interim", "feature_columns.txt"), "r")
        self.feature_list = f.read().split()

        f = open(os.path.join("data", "interim", "target_columns.txt"), "r")
        self.disease_list = f.read().split()

        self.df_best_model = pd.read_csv(os.path.join("data", "final", "ml_stats", "best_model.csv"))

        self.df_shap = pd.DataFrame(columns=["DISEASE"] + self.feature_list)


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def explain_all_diseases(self):
        for disease in self.disease_list:
            self.explain(disease)

        self.df_shap.to_csv(os.path.join("data", "final", "explainers", "shap_values.csv"), index=False)

        self.get_top_factors()
    

    def explain(self, disease):
        print(f"\nEXPLAINING {disease}\n")

        best_model = self.df_best_model.loc[self.df_best_model["disease"] == disease, "best_model"].values[0]
        model = self.load_model(disease, best_model)

        background_data, main_data = self.get_data(disease)

        explainer = shap.KernelExplainer(model.predict_proba, background_data,
                                         feature_names=self.feature_list)
        
        shap_values = explainer(main_data)[:, :, 1]
        overall_shap_values = np.abs(shap_values.values).mean(axis=0).tolist()

        shap_row = pd.DataFrame([[disease] + overall_shap_values], columns=self.df_shap.columns)
        self.df_shap = pd.concat([self.df_shap, shap_row], ignore_index=True)

        shap.plots.bar(shap_values)

        plt.savefig(os.path.join("data", "final", "explainers", f"explainer_{disease}.png"), dpi=300, bbox_inches='tight')
        plt.close()


    def get_data(self, disease):
        input, output = get_train_test_data(False, self.feature_list, disease)

        train_input, test_input, _, _ = train_test_split(
            input, output, test_size = 0.2, random_state = 10, stratify = output)
        
        background_data = shap.kmeans(train_input, 100)
        main_data = test_input[np.random.choice(len(test_input), 1000, replace=False)]

        return background_data, main_data


    def load_model(self, disease, best_model):
        if best_model == "xgboost":
            model = XGBClassifier()
            model.load_model(os.path.join("data", "final", "ml_models", "xgboost", f"{disease}.json"))
        elif best_model == "lr":
            model = joblib.load(os.path.join("data", "final", "ml_models", "logistic_regression", f"{disease}.pkl"))
        elif best_model == "nn":
            model = joblib.load(os.path.join("data", "final", "ml_models", "neural_network", f"{disease}.pkl"))
        return model
    

    def get_top_factors(self):
        if len(self.df_shap) == 0:
            self.df_shap = pd.read_csv(os.path.join("data", "final", "explainers", "shap_values.csv"))

        df = self.df_shap.copy().drop(columns=["GENHLTH", "PHYSHLTH", "POORHLTH"])

        factor_cols = df.columns[1:]

        def top_3_factors(row):
            return row[factor_cols].astype(float).nlargest(3).index.tolist()

        df['top_3_factors'] = df.apply(top_3_factors, axis=1)

        df_top_factors = pd.DataFrame(df['top_3_factors'].tolist(), 
                                      columns=['top1', 'top2', 'top3'])
        
        df_top_factors.insert(0, 'disease', df['DISEASE'])

        df_top_factors.to_csv(os.path.join("data", "final", "explainers", "top_factors.csv"), index=False)

        df_factor_list = df_top_factors.melt(id_vars='disease', value_vars=['top1','top2','top3'],
                                             value_name='factor')[['disease','factor']]
        
        df_factor_list.to_csv(os.path.join("data", "interim", "factor_list.csv"), index=False)



if __name__ == "__main__":
    ex = Explainer()
    ex.get_top_factors()