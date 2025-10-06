import os
import pandas as pd
import random

from datetime import (
    date,
    datetime, 
    timedelta, 
    time
)
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
        null_rows_indices = dataframe[column_name].isnull()
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

            safe_geography: tuple = safe_geographies[random_index]
            unexpected_geography: tuple = unexpected_geographies[random_index]

            if category.lower() == "spam":
                generated_data: tuple = unexpected_geography
            else:
                generated_data: tuple = safe_geography if random.randint(0, 10) > 0 else unexpected_geography

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
        regular_ip_ranges: List[str] = [
            "192.168.x.y",    # internal private LAN range
            "10.0.x.y",         # private subnet
            "172.16.x.y",        # private class B
            "203.0.113.y",  # telco NAT/test-net
            "124.13.x.y",     # Malaysia ISP allocation
        ]

        irregular_ip_ranges: List[str] = [
            "185.220.100.y",    # Tor exit node range
            "45.67.x.y",        # suspicious hosting provider block
            "198.51.100.y",     # reserved test-net (used here as "foreign DC" mock)
            "5.188.x.y",        # Eastern European DC range
            "37.120.x.y",       # VPN/hosting common net
        ]

        def generate_random_network_ip(category: str) -> str:
            random_index: int = random.randint(0, 4)
            third_octet: int = random.randint(0, 255)
            fourth_octet: int = random.randint(0, 255)
            
            translation_dict: dict = {
                "x" : str(third_octet),
                "y" : str(fourth_octet)
            }

            translation_table = str.maketrans(translation_dict)

            regular_ip: str = regular_ip_ranges[random_index]
            irregular_ip: str = irregular_ip_ranges[random_index]

            if (category.lower() == "spam"):
                generated_ip: str = irregular_ip
            else:
                generated_ip: str = regular_ip if random.randint(0, 10) > 0 else irregular_ip

            return generated_ip.translate(translation_table)
        
        if self.network_data_column not in dataframe:
            dataframe[self.network_data_column] = None

        if (dataframe[self.network_data_column].isnull().sum() > 0):
            null_rows: pd.DataFrame = self.__retrieve_null_rows(dataframe=dataframe, column_name=self.network_data_column)

            for index, row in null_rows.iterrows():
                generated_network_data: str = generate_random_network_ip(category=row.Category)
                dataframe.loc[index, self.network_data_column] = generated_network_data #type: ignore

        return dataframe
    
    def insert_temporal_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        start_datetime: datetime = datetime(2025, 7, 1, 9, 0, 0)
        end_datetime: datetime = datetime(2025, 9, 30, 18, 0, 0)

        def generate_random_date() -> date:
            days_span = (end_datetime.date() - start_datetime.date()).days
            offset_days = random.randint(0, days_span)
            return (start_datetime + timedelta(days=offset_days)).date()

        def generate_random_time(category: str) -> time:
            if category.lower() == "spam":
                hour: int = random.randint(9, 18)
            else:
                hour: int = random.randint(0, 5)
            minute: int = random.randint(0, 59)
            second: int = random.randint(0, 59)
            return time(hour, minute, second)
        
        if self.date_column not in dataframe:
            dataframe[self.date_column] = None

        if self.time_column not in dataframe:
            dataframe[self.time_column] = None

        if (dataframe[self.date_column].isnull().sum() > 0):
            null_rows: pd.DataFrame = self.__retrieve_null_rows(dataframe=dataframe, column_name=self.date_column)

            for index, _ in null_rows.iterrows():
                generated_date: date = generate_random_date()
                dataframe.loc[index, self.date_column] = generated_date #type: ignore

        if (dataframe[self.time_column].isnull().sum() > 0):
            null_rows: pd.DataFrame = self.__retrieve_null_rows(dataframe=dataframe, column_name=self.time_column)

            for index, row in null_rows.iterrows():
                generated_time = generate_random_time(category=row.Category)
                dataframe.loc[index, self.time_column] = generated_time #type: ignore
        return dataframe
    
feature_insertion = FeatureInsertion()