# <--- Imports --->

import os
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.machine_learning.roBERTa_method.torch_dataset import TorchDataset, PROCESSED_DATASET_PATH
from src.machine_learning.roBERTa_method.roberta_spam_classifier import RobertaSpamClassifier

# <--- Configurations --->

# --- Configuration ---
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
MODEL_SAVE_PATH: str = os.path.abspath("./src/machine_learning/roBERTa_method/trained_model/spam_classifier_model.pth")

if __name__ is "__main__":
    # Data loading and splitting
    print("\n--- Loading and Splitting Data ---\n")
    dataframe = pd.read_csv(PROCESSED_DATASET_PATH)

    train_dataframe, test_dataframe = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=42
    )

    print(f"\n--- Training on {len(train_dataframe)} samples, testing on {len(test_dataframe)} samples ---\n")

    train_dataset = TorchDataset(train_dataframe)
    test_dataset = TorchDataset(test_dataframe)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 

    # Initialize Model, Optimizer, Loss 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    number_of_features = train_dataset.number_of_tabular_features
    model = RobertaSpamClassifier(n_tabular_features=number_of_features)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("--- Starting Training ---")
for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tabular_features = batch['tabular_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, tabular_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch + 1} Avg. Training Loss: {total_loss / len(train_loader):.4f}")

    print("\n--- Training Complete ---\n")

    # --- 5. Evaluation Loop (NEW!) ---
    print("\n--- Starting Evaluation ---\n")
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tabular_features = batch['tabular_features'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask, tabular_features)
            
            _, predictions = torch.max(logits, dim=1)
            
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate and print the final accuracy
    accuracy = (total_correct / total_samples) * 100
    print(f"\n--- Evaluation Complete ---")
    print(f"Test Accuracy: {accuracy:.2f}%")

    # --- 6. Save the Final Model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")