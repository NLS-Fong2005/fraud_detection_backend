import os
import pandas as pd
import random

from typing import List

class FeatureInsertion:
    def __init__(self) -> None:
        self.mock_directory: str = os.path.abspath("./data/mock/")

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
        return dataframe

    def insert_network_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe
    
    def insert_temporal_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe
    
feature_insertion = FeatureInsertion()
