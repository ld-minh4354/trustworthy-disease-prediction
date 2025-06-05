import os, sys, time
import pandas as pd
import numpy as np
from utils.data_cleaning import data_cleaning
from utils.filter_rows_with_unsuitable_values import filter_rows_with_unsuitable_values


class DataProcessing:
    def __init__(self):
        self.add_project_folder_to_pythonpath()

        f = open(os.path.join("data", "interim", "feature_columns.txt"), "r")
        self.feature_columns = f.read().split()

        f = open(os.path.join("data", "interim", "target_columns.txt"), "r")
        self.target_columns = f.read().split()
    

    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)

    
    def main(self):
        self.read_raw_data()
        self.filter_unused_columns()
        self.convert_types_to_int()
        self.df = filter_rows_with_unsuitable_values(self.df)
        self.df = data_cleaning(self.df, self.target_columns)
        self.display_random_rows(50)
        self.calculate_class_imbalance()
        self.save_csv()

    
    def read_raw_data(self):
        start_time = time.time()
        self.df = pd.read_csv(os.path.join("data", "raw", "LLCP2023.csv"))
        end_time = time.time()

        print(f"Successfully read raw data in {end_time - start_time} seconds, containing {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

    
    def filter_unused_columns(self):
        self.df = self.df[self.feature_columns + self.target_columns]
        print(f"Successfully removing unused columns. There are now {self.df.shape[1]} columns.")
        

    def convert_types_to_int(self):
        is_integer = np.all(self.df.dropna().values == self.df.dropna().values.astype(int))
        print("All non-empty cells are integers:", is_integer)

        self.df = self.df.astype("Int64")

        all_int64 = (self.df.dtypes == "Int64").all()
        print("All columns are Int64:", all_int64)

    
    def display_random_rows(self, n_rows):
        print(self.df.iloc[:, :10].sample(n=n_rows))
        print(self.df.iloc[:, 10:20].sample(n=n_rows))
        print(self.df.iloc[:, 20:30].sample(n=n_rows))
        print(self.df.iloc[:, 30:40].sample(n=n_rows))
        print(self.df.iloc[:, 40:].sample(n=n_rows))

    
    def calculate_class_imbalance(self):
        self.df_imbalance = pd.DataFrame(columns=["disease", "ratio"])

        for disease in self.target_columns:
            count_0 = self.df[disease].eq(0).sum()
            count_1 = self.df[disease].eq(1).sum()

            new_row = pd.DataFrame([{"disease": disease, "ratio": count_0 / count_1}])
            self.df_imbalance = pd.concat([self.df_imbalance, new_row], ignore_index=True)

    
    def save_csv(self):
        self.df.to_csv(os.path.join("data", "interim", "ml_data_final.csv"), index=False)
        self.df_imbalance.to_csv(os.path.join("data", "interim", "ml_data_imbalance.csv"), index=False)



if __name__ == "__main__":
    dp = DataProcessing()
    dp.main()