import os
import pandas as pd


def get_train_test_data(testing, feature_columns, disease):
    df_ml_data = pd.read_csv(os.path.join("data", "interim", "ml_data_final.csv"))
    # print(f"Successfully read ml data, containing {df_ml_data.shape[0]} rows and {df_ml_data.shape[1]} columns.")

    if testing:
        df_ml_data = df_ml_data.head(1000)
        # print(f"Successfully filter rows for testing. Data now contains {df_ml_data.shape[0]} rows and {df_ml_data.shape[1]} columns.")

    input = df_ml_data[feature_columns].to_numpy()
    output = df_ml_data[[disease]].to_numpy().ravel()

    return input, output



if __name__ == "__main__":
    f = open(os.path.join("data", "interim", "feature_columns.txt"), "r")
    feature_columns = f.read().split()

    input, output = get_train_test_data(True, feature_columns, "BPHIGH6")

    print(input.shape)
    print(output.shape)