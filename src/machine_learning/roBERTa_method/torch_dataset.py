# src.machine_learning.roBERTa_method.torch_dataset

# <--- Imports --->
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

# <--- Configuration --->
PROCESSED_DATASET_PATH: str = os.path.abspath(path="./data/processed/processed_dataset.csv")
MESSAGE_COLUMN: str = "Message"
LABEL_COLUMN: str = "Category"

# <--- Torch Dataset --->
class TorchDataset(Dataset):
    def __init__(
            self,
            csv_file_path: str,
            tokeniser_name="roberta-base",
            max_length=128
    ):
        print("Loading Data...")
        self.dataframe = pd.read_csv(PROCESSED_DATASET_PATH)

        self.tokeniser = RobertaTokenizer.from_pretrained(tokeniser_name)
        self.max_length = max_length

        self.text_data = self.dataframe[MESSAGE_COLUMN].tolist()
        self.labels = self.dataframe[LABEL_COLUMN].values

        tabular_columns: list = [
            column for column in self.dataframe.columns
            if column not in [MESSAGE_COLUMN, LABEL_COLUMN]
        ]

        self.tabular_data = self.dataframe[tabular_columns]
        self.tabular_data = self.tabular_data.apply(pd.to_numeric, errors="coerce")
        self.tabular_data = self.tabular_data.fillna(0)
        self.tabular_data = self.tabular_data.values

        self.number_of_tabular_features = self.tabular_data.shape[1]

        print(f"\n--- Data Loaded. Found {len(self.dataframe)} rows. ---\n")
        print(f"\n Found {self.number_of_tabular_features} tabular features.\n")

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index) -> dict:
        message_content: str = self.text_data[index]
        tabular_features = self.tabular_data[index]
        label = self.labels[index]

        inputs = self.tokeniser(
            message_content,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tabular_features": torch.tensor(tabular_features, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long)
        }

    def data_loading(self):
        print("\n--- Testing New Torch Dataset ---\n")
        dataset = TorchDataset(csv_file_path=PROCESSED_DATASET_PATH)

        print("\n--- Testing __getitem__ for one item---\n")
        sample_item = dataset[5]

        print("Keys:", sample_item.keys())
        print("Shape of input_ids:", sample_item['input_ids'].shape)
        print("Shape of attention_mask:", sample_item['attention_mask'].shape)
        print("Shape of tabular_features:", sample_item['tabular_features'].shape)
        print("Label:", sample_item['label'])

        print("\n--- Testing Dataloader ---\n")
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True
        )

        one_batch = next(iter(data_loader))
    
        print("Keys in batch:", one_batch.keys())
        print("Batch shape of input_ids:", one_batch['input_ids'].shape)
        print("Batch shape of tabular_features:", one_batch['tabular_features'].shape)
        print("Batch shape of labels:", one_batch['label'].shape)
        
        print("\nSuccess! Our data pipeline is ready.")

torch_dataset = TorchDataset(csv_file_path=PROCESSED_DATASET_PATH)