import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.machine_learning.roBERTa_method.torch_dataset import torch_dataset
from transformers import RobertaModel

class RobertaSpamClassifier(nn.Module):
    def __init__(
            self,
            number_of_tabular_features,
            number_of_classes=2
    ):
        print(f"Numebr of Tabular Features: {number_of_tabular_features}")
        super(RobertaSpamClassifier, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.roberta_output_size = 768

        self.tabular_net = nn.Sequential(
            nn.Linear(number_of_tabular_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.tabular_output_size = 64

        combined_size = self.roberta_output_size + self.tabular_output_size

        self.classifier_head = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, number_of_classes)
        )

    def forward(
            self,
            input_ids,
            attention_mask,
            tabular_features
    ):
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_embedding = roberta_output.pooler_output

        tabular_embedding = self.tabular_net(tabular_features)

        combined_embedding = torch.cat(
            (text_embedding, tabular_embedding),
            dim=1
        )

        logits = self.classifier_head(combined_embedding)

        return logits
    
def run_model_testing():
    print("\n--- Testing Model Architecture ---\n")

    print("\n--- Loading Dataset to get Model Parameters ---\n")
    temp_dataset = torch_dataset

    number_of_features = temp_dataset.number_of_tabular_features
    print(f"Found {number_of_features} tabular features.")

    print("\n--- Initialising Model ---\n")
    model = RobertaSpamClassifier(number_of_tabular_features=number_of_features)

    data_loader = DataLoader(temp_dataset, batch_size=4)
    one_batch = next(iter(data_loader))

    input_ids = one_batch['input_ids']
    attention_mask = one_batch['attention_mask']
    tabular_features = one_batch['tabular_features']

    print(f"Input text shape (batch_size, max_len): {input_ids.shape}")
    print(f"Input tabular shape (batch_size, n_features): {tabular_features.shape}")

    model.eval()

    with torch.no_grad():
        print("Performing a forward pass...")

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tabular_features=tabular_features
        )
    
    print("\n--- Model Output (Logits) ---")
    print(logits)
    print("Output shape (batch_size, n_classes):", logits.shape)
    
    if logits.shape == (4, 2):
        print("\nSuccess! Our custom model is built and accepts data correctly.")
    else:
        print("\nError: Output shape is incorrect. Please check model definitions.")
