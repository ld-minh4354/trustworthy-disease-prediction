import os, sys
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils import ResNet, get_dataloader, training, evaluation


class ModelTraining:
    def __init__(self, testing=False):
        self.testing = testing
        
        self.add_project_folder_to_pythonpath()

        f = open(os.path.join("data", "interim", "feature_columns.txt"), "r")
        self.feature_columns = f.read().split()

        f = open(os.path.join("data", "interim", "target_columns.txt"), "r")
        self.target_columns = f.read().split()

        self.df_imbalance = pd.read_csv(os.path.join("data", "interim", "ml_data_imbalance.csv"))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on {self.device}.")

    
    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)
    

    def ml_training(self):
        for disease in self.target_columns:
            self.ml_training_one_disease(disease)


    def ml_training_one_disease(self, disease):
        input_dim = len(self.feature_columns)
        output_dim = 2

        train_loader, test_loader = get_dataloader(self.testing, self.feature_columns, disease)

        model = ResNet(input_dim, output_dim).to(self.device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

        ratio_value = self.df_imbalance.loc[self.df_imbalance["disease"] == disease, "ratio"].values
        ratio_value = float(ratio_value[0])

        print(f"Training for disease: {disease}, ratio value: {ratio_value}")

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, ratio_value]))
        optimizer = optim.Adam(model.parameters(), lr = 0.001)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        num_epochs = 50

        training(train_loader, test_loader,
                 model, criterion, optimizer, scheduler, num_epochs,
                 self.device, disease)


if __name__ == "__main__":
    mt = ModelTraining(testing=False)
    mt.ml_training()
