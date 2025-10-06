#IMPORTS GO HERE
from src.core.mock_feature_engineering import feature_insertion

import pandas as pd
import os

#GLOBAL VARIABLES GO HERE

#FUNCTIONS GO HERE
def main():
    dataset_path = os.path.abspath("./data/raw/spam.csv")
    dataframe: pd.DataFrame = pd.read_csv(dataset_path)

    if "Unnamed: 0" in dataframe:
        dataframe = dataframe.drop(columns=["Unnamed: 0"], axis=1)

    # dataframe = feature_insertion.insert_temporal_data(dataframe=dataframe)
    # dataframe = feature_insertion.insert_network_data(dataframe=dataframe)
    dataframe = feature_insertion.insert_geographical_data(dataframe=dataframe)
    # dataframe = feature_insertion.increasing_spam_frequency(dataframe=dataframe)

    # feature_insertion.export_to_path(dataframe=dataframe)

if __name__ == '__main__':
    main()