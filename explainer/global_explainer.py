from model import ResNet
import pandas as pd
import os
import torch
import torch.nn.functional as F
import shap
import numpy as np
import matplotlib.pyplot as plt


class GlobalExplainer:
    def __init__(self):
        self.load_data()

        f = open(os.path.join("data", "interim", "target_columns.txt"), "r")
        self.target_columns = f.read().split()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")


    def load_data(self):
        df = pd.read_csv(os.path.join("data", "interim", "ml_data_final.csv"))
        self.feature_names = df.columns[:38].tolist()

        data = df.to_numpy()[:, :38]

        sample_size = 5
        idx = np.random.choice(len(data), sample_size, replace=False)
        self.data_sample = data[idx]

        self.background_data = shap.kmeans(data, 50)


    def load_model(self, disease):
        self.model = ResNet(self.data_sample.shape[1], 2).to(self.device)
        self.model.load_state_dict(torch.load(
            os.path.join("data", "final", "ml_models", f"final_{disease}.pth"),
            map_location=self.device
        ))
        self.model.eval()


    def model_wrapper(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=1)
            return probs[:, 1].detach().cpu().numpy()
        

    def explainer(self, disease):
        print(f"Current disease: {disease}\n")
        self.load_model(disease)
        self.model_explainer = shap.KernelExplainer(model=self.model_wrapper, 
                                                    data=self.background_data,
                                                    feature_names=self.feature_names)

        shap_values = self.model_explainer(self.data_sample)
        shap.plots.beeswarm(shap_values)

        plt.savefig(os.path.join("data", "final", "explainer", f"global_explainer_{disease}_test.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Disease {disease} done.")
        print("--------------------------------------------------")

    
    def explainer_all(self):
        for disease in self.target_columns:
            self.explainer(disease)


if __name__ == "__main__":
    explainer = GlobalExplainer()
    explainer.explainer("CHCOCNC1")
