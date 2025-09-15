import os, sys, time, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import ConvergenceWarning
from get_train_test_data import get_train_test_data
import joblib


from sklearn.neighbors import KNeighborsClassifier


class KNNModel:
    def __init__(self):
        self.add_project_folder_to_pythonpath()

        f = open(os.path.join("data", "interim", "feature_columns.txt"), "r")
        self.feature_columns = f.read().split()

        f = open(os.path.join("data", "interim", "target_columns.txt"), "r")
        self.target_columns = f.read().split()

        self.df_stats = pd.DataFrame(columns = [
            "disease",
            "train_accuracy", "train_precision", "train_recall", "train_f1", "train_loss", "train_roc",
            "test_accuracy", "test_precision", "test_recall", "test_f1", "test_loss", "test_roc",
            "n_neighbors", "weights"
        ])

        self.df_imbalance = pd.read_csv(os.path.join("data", "interim", "ml_data_imbalance.csv"))

        warnings.filterwarnings("ignore", category = ConvergenceWarning)


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def main(self):
        self.train_all_diseases()
        self.save_stats()


    def train_all_diseases(self):
        for disease in self.target_columns:
            self.train_disease(disease)
    

    def train_disease(self, disease):
        print(f"\nTRAINING KNN FOR {disease}")

        start_time = time.time()

        input, output = get_train_test_data(False, self.feature_columns, disease)

        train_input, test_input, train_output, test_output = train_test_split(
            input, output, test_size = 0.2, random_state = 10, stratify = output)
        
        # smote = SMOTE(random_state = 100)
        # train_input, train_output = smote.fit_resample(train_input, train_output)
        
        param_grid = {"n_neighbors": [50, 100, 150, 200, 300, 500],
                      "weights": ["uniform", "distance"]}

        ratio_value = self.df_imbalance.loc[self.df_imbalance["disease"] == disease, "ratio"].values
        ratio_value = float(ratio_value[0])

        lr = KNeighborsClassifier(n_jobs = 4)
        grid = GridSearchCV(lr, param_grid, scoring = "f1")
        grid.fit(train_input, train_output)

        param = grid.best_params_
        print("Best param:", param)

        model = grid.best_estimator_
        joblib.dump(model, os.path.join("data", "final", "ml_models", "knn", f"{disease}.pkl"))

        train_pred = model.predict(train_input)
        test_pred = model.predict(test_input)

        train_proba = model.predict_proba(train_input)[:, 1]
        test_proba = model.predict_proba(test_input)[:, 1]

        print("\nTraining stats")
        train_accuracy, train_precision, train_recall, train_f1, train_loss, train_roc = self.get_stats(train_output, train_pred, train_proba)

        print("\nTesting stats")
        test_accuracy, test_precision, test_recall, test_f1, test_loss, test_roc = self.get_stats(test_output, test_pred, test_proba)

        stats = {"disease": disease,
                 "train_accuracy": train_accuracy,
                 "train_precision": train_precision,
                 "train_recall": train_recall,
                 "train_f1": train_f1,
                 "train_loss": train_loss,
                 "train_roc": train_roc,
                 "test_accuracy": test_accuracy,
                 "test_precision": test_precision,
                 "test_recall": test_recall,
                 "test_f1":test_f1,
                 "test_loss": test_loss,
                 "test_roc": test_roc,
                 "n_neighbors": param["n_neighbors"],
                 "weights": param["weights"]}
        
        self.df_stats.loc[len(self.df_stats)] = stats

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"\nTotal run time: {elapsed:.2f} seconds")


    def get_stats(self, output, pred, proba):
        accuracy = accuracy_score(output, pred)
        precision = precision_score(output, pred)
        recall = recall_score(output, pred)
        f1 = f1_score(output, pred)
        roc = roc_auc_score(output, proba)
        loss = mean_squared_error(output, proba)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("ROC:", roc)
        print("Loss:", loss)

        return accuracy, precision, recall, f1, loss, roc


    def save_stats(self):
        self.df_stats.to_csv(os.path.join("data", "final", "ml_stats", "knn.csv"), index=False)




if __name__ == "__main__":
    model = KNNModel()
    model.main()

