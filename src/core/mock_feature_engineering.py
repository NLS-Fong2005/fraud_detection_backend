import os
import pandas as pd
import random

from typing import List

class FeatureInsertion:
    def __init__(self) -> None:
        self.mock_directory: str = os.path.abspath("./data/mock/")
        self.geographical_data_column: str = "Source_Location"
        self.network_data_column: str = "Source_IP"
        self.date_column: str = "Sent_Date"
        self.time_column: str = "Sent_Time"

    def __retrieve_null_rows(
            self, 
            dataframe: pd.DataFrame, 
            column_name: str
    ) -> pd.DataFrame:
        null_rows_indices = dataframe[self.geographical_data_column].isnull()
        null_rows: pd.DataFrame = dataframe[null_rows_indices]

        return null_rows

    def export_to_path(self, dataframe: pd.DataFrame) -> None:
        dataset_path: str = os.path.abspath(f"{self.mock_directory}/mock_dataset.csv")
        try:
            dataframe.to_csv(path_or_buf=dataset_path, index=False)
            print(f"Successfully saved to output csv.")
        except FileExistsError:
            os.remove(dataset_path)
            dataframe.to_csv(path_or_buf=dataset_path, index=False)
            print(f"Successfully saved to output csv.")
        except FileNotFoundError:
            print("Directory does not exist. Please ensure mock directory is made under data.")
        except Exception as e:
            print(f"Something went wrong: {e}")

    def increasing_spam_frequency(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frequency_of_spam: int = random.randint(1, 10)
        random_count: int = random.randint(1, 10)

        def retrieve_spam_rows() -> List[int]:
            spam_rows = dataframe.loc[dataframe["Category"] == "spam"].index
            return list(spam_rows)

        random_spam_indices: List[int] = random.sample(retrieve_spam_rows(), random_count)

        duplicated_rows: pd.DataFrame = pd.concat([dataframe.loc[random_spam_indices]] * frequency_of_spam, ignore_index=True)
        final_dataframe: pd.DataFrame = pd.concat([dataframe, duplicated_rows], ignore_index=True)
        
        return final_dataframe
    
    def insert_geographical_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        safe_geographies = [
            ("Malaysia", "Kuala Lumpur"),
            ("Malaysia", "Selangor"),
            ("Singapore", "Central"),
            ("Indonesia", "DKI Jakarta"),
            ("Thailand", "Bangkok"),
        ]

        unexpected_geographies = [
            ("United States", "Virginia"),        # common DC region
            ("Netherlands", "North Holland"),     # hosting-heavy region
            ("Germany", "Hesse"),                 # Frankfurt DC hub
            ("Hong Kong", "Hong Kong"),           # cross-border DC/hosting
            ("Romania", "Bucharest"),
        ]
        
        def generate_random_geographical_data(category: str) -> str:
            random_index: int = random.randint(0, 4)
            if category.lower() == "spam":
                generated_data: tuple = unexpected_geographies[random_index]
            else:
                generated_data: tuple = safe_geographies[random_index] if random.randint(0, 10) > 0 else unexpected_geographies[random_index]
            return str(generated_data)

        if self.geographical_data_column not in dataframe:
            dataframe[self.geographical_data_column] = None
        
        if (dataframe[self.geographical_data_column].isnull().sum() > 0):
            null_rows: pd.DataFrame = self.__retrieve_null_rows(dataframe=dataframe, column_name=self.geographical_data_column)

            for index, row in null_rows.iterrows():
                generated_geographical_data: str = generate_random_geographical_data(category=row.Category)
                dataframe.loc[index, self.geographical_data_column] = generated_geographical_data #type: ignore

        return dataframe

    def insert_network_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe
    
    def insert_temporal_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe
    
feature_insertion = FeatureInsertion()
