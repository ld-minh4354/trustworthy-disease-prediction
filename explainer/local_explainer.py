from model import ResNet
import pandas as pd
import os
import torch
import torch.nn.functional as F
import shap
import numpy as np
import matplotlib.pyplot as plt


class LocalExplainer:
    def __init__(self):
        self.load_data()
        self.load_model()
        self.model_explainer = shap.KernelExplainer(self.model_wrapper, self.background_data)


    def load_data(self):
        df = pd.read_csv(os.path.join("data", "interim", "ml_data_final.csv"))
        self.feature_names = df.columns[:38].tolist()

        data = df.to_numpy()[:, :38]

        self.data_point = data[2:3]
        self.background_data = shap.kmeans(data, 100)

    
    def load_model(self):
        self.model = ResNet(len(self.data_point[0]), 2)
        self.model.load_state_dict(torch.load(os.path.join("data", "final", "ml_models", "final_DIABETE4.pth")))
        self.model.eval()

    
    def model_wrapper(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=1)
            return probs[:, 1].unsqueeze(1).numpy()

    
    def explainer(self):
        shap_values = self.model_explainer.shap_values(self.data_point)

        shap.initjs()
        shap.force_plot(
            self.model_explainer.expected_value,
            shap_values[0].T[0],
            self.data_point[0],
            feature_names=self.feature_names,
            matplotlib=True)

        plt.savefig(os.path.join("data", "final", "explainer", "local_explainer.png"), dpi=300, bbox_inches='tight')
        plt.close()
        


if __name__ == "__main__":
    explainer = LocalExplainer()
    explainer.explainer()