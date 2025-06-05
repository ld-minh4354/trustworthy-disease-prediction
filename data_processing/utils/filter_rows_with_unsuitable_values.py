import pandas as pd


def filter_rows_with_unsuitable_values(df):
        # Gender
        df = df[df["SEXVAR"].isin([1, 2])]

        # General health
        df = df[df["GENHLTH"].isin([1, 2, 3, 4, 5])]
        df = df[((df["PHYSHLTH"] >= 1) & (df["PHYSHLTH"] <= 30)) | (df["PHYSHLTH"] == 88)]
        df = df[((df["MENTHLTH"] >= 1) & (df["MENTHLTH"] <= 30)) | (df["MENTHLTH"] == 88)]
        df = df[((df["POORHLTH"] >= 1) & (df["POORHLTH"] <= 30)) | (df["POORHLTH"] == 88) | ((df["PHYSHLTH"] == 88) & (df["MENTHLTH"] == 88))]

        # Healthcare access
        df = df[df["PERSDOC3"].isin([1, 2, 3])]
        df = df[df["MEDCOST1"].isin([1, 2])]
        df = df[df["CHECKUP1"].isin([1, 2, 3, 4, 8])]

        # Exercising
        df = df[df["EXERANY2"].isin([1, 2])]

        # Most often exercise method
        df = df[((df["EXRACT12"] >= 1) & (df["EXRACT12"] <= 10)) | (df["EXERANY2"] == 2)]
        df = df[((df["EXEROFT1"] >= 101) & (df["EXEROFT1"] <= 299)) | (df["EXERANY2"] == 2)]
        df = df[((df["EXERHMM1"] >= 1) & (df["EXERHMM1"] <= 959) & (df["EXERHMM1"] != 777)) | (df["EXERANY2"] == 2)]

        # Second often exercise method
        df = df[((df["EXRACT22"] >= 1) & (df["EXRACT22"] <= 10)) | (df["EXERANY2"] == 2)]
        df = df[((df["EXEROFT2"] >= 101) & (df["EXEROFT2"] <= 299)) | (df["EXERANY2"] == 2)]
        df = df[((df["EXERHMM2"] >= 1) & (df["EXERHMM2"] <= 959) & (df["EXERHMM2"] != 777)) | (df["EXERANY2"] == 2)]

        # Strength training
        df = df[((df["STRENGTH"] >= 101) & (df["STRENGTH"] <= 299)) | (df["STRENGTH"] == 888)]

        # Diseases
        df = df[df["BPHIGH6"].isin([1, 2, 3, 4])]
        df = df[df["TOLDHI3"].isin([1, 2])]
        df = df[df["CVDINFR4"].isin([1, 2])]
        df = df[df["CVDCRHD4"].isin([1, 2])]
        df = df[df["CVDSTRK3"].isin([1, 2])]
        df = df[df["ASTHMA3"].isin([1, 2])]
        df = df[df["CHCSCNC1"].isin([1, 2])]
        df = df[df["CHCOCNC1"].isin([1, 2])]
        df = df[df["CHCCOPD3"].isin([1, 2])]
        df = df[df["ADDEPEV3"].isin([1, 2])]
        df = df[df["CHCKDNY2"].isin([1, 2])]
        df = df[df["HAVARTH4"].isin([1, 2])]
        df = df[df["DIABETE4"].isin([1, 2, 3, 4])]

        # Personal info
        df = df[df["MARITAL"].isin([1, 2, 3, 4, 5, 6])]
        df = df[df["EDUCA"].isin([1, 2, 3, 4, 5, 6])]
        df = df[df["RENTHOM1"].isin([1, 2, 3])]
        df = df[df["EMPLOY1"].isin([1, 2, 3, 4, 5, 6, 7, 8])]

        # Height and weight
        df = df[((df["WEIGHT2"] >= 50) & (df["WEIGHT2"] <= 776)) | ((df["WEIGHT2"] >= 9023) & (df["WEIGHT2"] <= 9352))]
        df = df[((df["HEIGHT3"] >= 200) & (df["HEIGHT3"] <= 711)) | ((df["HEIGHT3"] >= 9061) & (df["HEIGHT3"] <= 9998))]

        # Other conditions
        df = df[df["DEAF"].isin([1, 2])]
        df = df[df["BLIND"].isin([1, 2])]
        df = df[df["DECIDE"].isin([1, 2])]
        df = df[df["DIFFWALK"].isin([1, 2])]
        df = df[df["DIFFDRES"].isin([1, 2])]
        df = df[df["DIFFALON"].isin([1, 2])]

        # Smoking
        df = df[df["SMOKE100"].isin([1, 2])]
        df = df[(df["SMOKDAY2"].isin([1, 2, 3])) | (df["SMOKE100"] == 2)]
        df = df[df["USENOW3"].isin([1, 2, 3])]
        df = df[df["ECIGNOW2"].isin([1, 2, 3, 4])]

        # Drinking
        df = df[((df["ALCDAY4"] >= 101) & (df["ALCDAY4"] <= 299)) | (df["ALCDAY4"] == 888)]
        df = df[((df["AVEDRNK3"] >= 1) & (df["AVEDRNK3"] <= 76)) | (df["AVEDRNK3"] == 88) | (df["ALCDAY4"] == 888)]
        df = df[((df["DRNK3GE5"] >= 1) & (df["DRNK3GE5"] <= 76)) | (df["DRNK3GE5"] == 88) | (df["ALCDAY4"] == 888)]

        # COVID-19
        df = df[df["COVIDPO1"].isin([1, 2])]
        df = df[(df["COVIDSM1"].isin([1, 2])) | (df["COVIDPO1"] == 2)]
        df = df[(df["COVIDACT"].isin([1, 2])) | (df["COVIDPO1"] == 2) | (df["COVIDSM1"] == 2)]

        df = df.reset_index(drop=True)

        print(f"Filtering rows successfully. There are now {df.shape[0]} rows.")
        return df