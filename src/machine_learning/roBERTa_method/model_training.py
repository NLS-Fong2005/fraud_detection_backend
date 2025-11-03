# <--- Imports --->

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.machine_learning.roBERTa_method.torch_dataset import torch_dataset
from src.machine_learning.roBERTa_method.roberta_spam_classifier import RobertaSpamClassifier

# <--- Configurations --->
EPOCHS: int = 3
BATCH_SIZE: int = 16
LEARNING_RATE: int = 1e-5

MODEL_SAVE_PATH: str = os.path.abspath("./src/machine_learning/roBERTa_method/trained_model/spam_classifier_model.pth")

def train():
    print("\n--- Starting Training ---\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- Loading Dataset ---\n")
    full_dataset = torch_dataset

    data_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("\n--- Initialising Model ---\n")
    number_of_features = full_dataset.number_of_tabular_features

    model = RobertaSpamClassifier(number_of_tabular_features=number_of_features)

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/ {EPOCHS}---\n")

        model.train()

        total_loss = 0.0

        for batch in tqdm(data_loader, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tabular_features = batch["tabular_features"].to(device)
            labels = batch["label"].to(device)

            optimiser.zero_grad()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tabular_features=tabular_features
            )

            loss = criterion(logits, labels)

            loss.backward()

            optimiser.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1} Average Loss: {average_loss:.4f}")
    
    print("\n--- Training Complete ---\n")
    
    try:
        print("\n--- Saving Model ---\n")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Model failed to save: {e}")

if __name__ == "__main__":
    train()