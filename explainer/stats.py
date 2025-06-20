import os
import pandas as pd
import numpy as np


class Stats:
    def __init__(self):
        self.df = pd.read_csv(os.path.join("data", "interim", "ml_data_final.csv"))

        f = open(os.path.join("data", "interim", "target_columns.txt"), "r")
        self.target_columns = f.read().split()

    
    def stats_calculation(self, field):
        result = pd.DataFrame()

        for col in self.target_columns:
            percent = (
                self.df.groupby(field)[col]
                .apply(lambda x: (x == 1).mean() * 100)
                .reset_index(name=f"{col}")
            )
            if result.empty:
                result = percent
            else:
                result = result.merge(percent, on=field)
        
        return result.round(2)


    def employment(self):
        self.df["EMPLOY1"] = np.where(self.df["EMPLOY1"].between(1, 5), 1, self.df["EMPLOY1"])

        result = self.stats_calculation("EMPLOY1")
        
        print(result)
        result.to_csv(os.path.join("data", "final", "stats", "employment.csv"), index=False)


    def marital_status(self):
        self.df["MARITAL"] = self.df["MARITAL"].replace({1: 1, 2: 2, 3: 3, 4: 1, 5: 1, 6: 1})

        result = self.stats_calculation("MARITAL")
        
        print(result)
        result.to_csv(os.path.join("data", "final", "stats", "marital_status.csv"), index=False)

    
    def sex(self):
        result = self.stats_calculation("SEXVAR")
        
        print(result)
        result.to_csv(os.path.join("data", "final", "stats", "sex.csv"), index=False)


    def education(self):
        result = self.stats_calculation("EDUCA")
        
        print(result)
        result.to_csv(os.path.join("data", "final", "stats", "education.csv"), index=False)



if __name__ == "__main__":
    stats = Stats()
    stats.education()
