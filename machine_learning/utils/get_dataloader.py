import os
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split


def get_dataloader(testing, feature_columns, disease):
    df_ml_data = pd.read_csv(os.path.join("data", "interim", "ml_data_final.csv"))
    print(f"Successfully read ml data, containing {df_ml_data.shape[0]} rows and {df_ml_data.shape[1]} columns.")

    if testing:
        df_ml_data = df_ml_data.head(10000)
        print(f"Successfully filter rows for testing. Data now contains {df_ml_data.shape[0]} rows and {df_ml_data.shape[1]} columns.")

    df_feature = df_ml_data[feature_columns]
    df_target = df_ml_data[[disease]]

    torch_feature = torch.tensor(df_feature.values, dtype=torch.float32)
    torch_target = torch.tensor(df_target.values, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(torch_feature, torch_target, test_size=0.2, random_state=1)

    print("Successfully create training and testing data:")
    print(f"Training feature: {X_train.shape}")
    print(f"Training target: {y_train.shape}")
    print(f"Testing feature: {X_test.shape}")
    print(f"Testing target: {y_test.shape}")

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Successfully create training and testing dataloader, containing {len(train_loader.dataset)} and {len(test_loader.dataset)} samples respectively.")

    return train_loader, test_loader