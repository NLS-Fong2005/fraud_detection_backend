import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# <--- Configuration --->

class FeatureEngineering:
    def __init__(self):
        self.LABEL_COLUMN: str = "Category"
        self.TEMPORAL_COLUMN: str = "Sent_Time"
        self.NETWORK_DATA: str = "Source_IP"
        self.GEOGRAPHICAL_DATA: str = "Source_Location"

    def __export_to_path__(self, dataframe: pd.DataFrame):
        PROCESSED_DIRECTORY: str = os.path.abspath("./data/processed")
        PROCESSED_DATASET_PATH: str = os.path.abspath(f"{PROCESSED_DIRECTORY}/processed_dataset.csv")

        try:
            dataframe.to_csv(path_or_buf=PROCESSED_DATASET_PATH, index=False)
            print(f"Successfully saved to {PROCESSED_DATASET_PATH}")
        except FileExistsError:
            os.remove(PROCESSED_DATASET_PATH)
            dataframe.to_csv(path_or_buf=PROCESSED_DATASET_PATH, index=False)
            print(f"Successfully saved to {PROCESSED_DATASET_PATH}")
        except FileNotFoundError:
            print(f"{PROCESSED_DIRECTORY} does not exist! Please ensure processed directory is made under {os.path.abspath("./data/")}")
        except Exception as e:
            print(f"Something went wrong: {e}")

    def feature_engineer_dataset(self, dataframe: pd.DataFrame):
        try:
            print(f"Original Dataset Shape: {dataframe.shape}")
            print("\n--- Original Columns and Data Types ---\n")
            dataframe.info()

            print(f"\n--- Processing Temporal Data ---\n")
            dataframe = dataframe.drop("Sent_Date", axis=1)

            dataframe["Sent_Time"] = pd.to_datetime(dataframe["Sent_Time"])
            dataframe["Sent_Hour"] = dataframe["Sent_Time"].dt.hour

            print("\n--- Processing Network Data ---\n")
            ip_counts = dataframe[self.NETWORK_DATA].value_counts()
            dataframe["IP_Frequency"] = dataframe[self.NETWORK_DATA].map(ip_counts)

            dataframe["IP_Frequency"] = dataframe["IP_Frequency"].fillna(0)

            print("\n--- Processing Geographical Column ---\n")
            country_state = dataframe[self.GEOGRAPHICAL_DATA].str.strip("()").str.split(", ", expand=True, n=1)

            dataframe["Country"] = country_state[0]
            dataframe["State"] = country_state[1]

            dataframe["Country"] = dataframe["Country"].fillna('Unknown').replace('', 'Unknown')
            dataframe["State"] = dataframe["State"].fillna('Unknown').replace('', 'Unknown')

            one_hot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
            encoded_data_array = one_hot_encoder.fit_transform(dataframe[["Country", "State"]])
            new_column_names = one_hot_encoder.get_feature_names_out(["Country", "State"])

            one_encoded_df = pd.DataFrame(
                encoded_data_array, 
                columns=new_column_names, 
                index=dataframe.index  # This is important to align the rows correctly
            )

            dataframe = dataframe.drop(columns=["Country", "State"])

            dataframe = pd.concat([dataframe, one_encoded_df], axis=1)

            label_encoder = LabelEncoder()

            dataframe[self.LABEL_COLUMN] = label_encoder.fit_transform(dataframe[self.LABEL_COLUMN])

            print("\n--- New Dataframe ---\n")
            print(dataframe)

            print("\n--- Dataframe Info ---\n")
            dataframe.info()

            print("\n--- Exporting Dataset ---\n")
            self.__export_to_path__(dataframe=dataframe)
        except KeyError as e:
            print(f"Column not Found: {e}")

feature_engineering = FeatureEngineering()