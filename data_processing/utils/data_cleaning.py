import pandas as pd
import numpy as np


def data_cleaning(df, target_columns):
        # General health
        df.loc[df["PHYSHLTH"] == 88, "PHYSHLTH"] = 0
        df.loc[df["MENTHLTH"] == 88, "MENTHLTH"] = 0
        df.loc[((df["PHYSHLTH"] == 0) & (df["MENTHLTH"] == 0)) | (df["POORHLTH"] == 88), "POORHLTH"] = 0

        # Healthcare access
        df.loc[df["CHECKUP1"] == 8, "CHECKUP1"] = 5

        # Most often exercise method
        df.loc[df["EXERANY2"] == 2, "EXRACT12"] = 0
        
        df.loc[df["EXERANY2"] == 2, "EXEROFT1"] = 0
        df.loc[(df["EXEROFT1"] >= 201) & (df["EXEROFT1"] <= 299), "EXEROFT1"] = df["EXEROFT1"] - 200
        df.loc[(df["EXEROFT1"] >= 101) & (df["EXEROFT1"] <= 199), "EXEROFT1"] = (df["EXEROFT1"] - 100) * 4
        df.loc[df["EXEROFT1"] > 99, "EXEROFT1"] = 99

        df.loc[df["EXERANY2"] == 2, "EXERHMM1"] = 0
        df.loc[df["EXERANY2"] == 1, "EXERHMM1"] = (df["EXERHMM1"] // 100) * 60 + df["EXERHMM1"] % 100

        # Second most often exercise method
        df.loc[df["EXERANY2"] == 2, "EXRACT22"] = 0
        
        df.loc[df["EXERANY2"] == 2, "EXEROFT2"] = 0
        df.loc[(df["EXEROFT2"] >= 201) & (df["EXEROFT2"] <= 299), "EXEROFT2"] = df["EXEROFT2"] - 200
        df.loc[(df["EXEROFT2"] >= 101) & (df["EXEROFT2"] <= 199), "EXEROFT2"] = (df["EXEROFT2"] - 100) * 4
        df.loc[df["EXEROFT2"] > 99, "EXEROFT2"] = 99

        df.loc[df["EXERANY2"] == 2, "EXERHMM2"] = 0
        df.loc[df["EXERANY2"] == 1, "EXERHMM2"] = (df["EXERHMM2"] // 100) * 60 + df["EXERHMM2"] % 100

        # Strength training
        df.loc[df["STRENGTH"] == 888, "STRENGTH"] = 0
        df.loc[(df["STRENGTH"] >= 201) & (df["STRENGTH"] <= 299), "STRENGTH"] = df["STRENGTH"] - 200
        df.loc[(df["STRENGTH"] >= 101) & (df["STRENGTH"] <= 199), "STRENGTH"] = (df["STRENGTH"] - 100) * 4
        df.loc[df["STRENGTH"] > 99, "STRENGTH"] = 99

        # Diseases
        df.loc[df["BPHIGH6"] >= 2, "BPHIGH6"] = 2
        df.loc[df["DIABETE4"] >= 2, "DIABETE4"] = 2

        # Weights and Heights
        df.loc[df["WEIGHT2"] < 1000, "WEIGHT2"] = np.ceil(df["WEIGHT2"] * 0.45359237)
        df.loc[df["WEIGHT2"] > 1000, "WEIGHT2"] = df["WEIGHT2"] - 9000

        df.loc[df["HEIGHT3"] < 1000, "HEIGHT3"] = np.floor(((df["HEIGHT3"] // 100) * 12 + df["HEIGHT3"] % 100) * 2.54)
        df.loc[df["HEIGHT3"] > 1000, "HEIGHT3"] = df["HEIGHT3"] - 9000

        # Smoking
        df.loc[df["SMOKE100"] == 2, "SMOKDAY2"] = 3

        # Drinking
        df.loc[df["ALCDAY4"] == 888, "ALCDAY4"] = 0
        df.loc[(df["ALCDAY4"] >= 201) & (df["ALCDAY4"] <= 299), "ALCDAY4"] = df["ALCDAY4"] - 200
        df.loc[(df["ALCDAY4"] >= 101) & (df["ALCDAY4"] <= 199), "ALCDAY4"] = (df["ALCDAY4"] - 100) * 4
        df.loc[df["ALCDAY4"] > 99, "ALCDAY4"] = 99

        df.loc[(df["ALCDAY4"] == 0) | (df["AVEDRNK3"] == 88), "AVEDRNK3"] = 0
        df.loc[(df["ALCDAY4"] == 0) | (df["DRNK3GE5"] == 88), "DRNK3GE5"] = 0

        # COVID-19
        df.loc[df["COVIDPO1"] == 2, "COVIDSM1"] = 2
        df.loc[df["COVIDSM1"] == 2, "COVIDACT"] = 3

        # Change target data
        df[target_columns] = 2 - df[target_columns]

        print("Cleaning data successfully.")
        return df