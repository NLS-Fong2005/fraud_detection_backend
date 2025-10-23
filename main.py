# <-- Imports -->
from src.core.mock_feature_generation import feature_insertion
from src.machine_learning.roBERTa_method.roBERTa_data_training import roberta_model
from src.machine_learning.feature_engineering import feature_engineering

import pandas as pd
import os

# <-- Functions -->

def generate_mock_dataset(dataframe: pd.DataFrame) -> None:
    if "Unnamed: 0" in dataframe:
        dataframe = dataframe.drop(columns=["Unnamed: 0"], axis=1)

    dataframe = feature_insertion.insert_temporal_data(dataframe=dataframe)
    dataframe = feature_insertion.insert_network_data(dataframe=dataframe)
    dataframe = feature_insertion.insert_geographical_data(dataframe=dataframe)
    dataframe = feature_insertion.increasing_spam_frequency(dataframe=dataframe)

    print(f"Final Dataframe:\n{dataframe}")

    feature_insertion.export_to_path(dataframe=dataframe)

def main():
    """Main Function"""
    dataset_path = os.path.abspath("./data/mock/mock_dataset.csv")
    dataframe: pd.DataFrame = pd.read_csv(dataset_path)

    # generate_mock_dataset(dataframe=dataframe)
    # roberta_model.data_loading()
    feature_engineering.feature_engineer_dataset(dataframe)

if __name__ == '__main__':
    main()