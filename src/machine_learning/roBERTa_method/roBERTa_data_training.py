import os
import pandas as pd
from transformers import RobertaTokenizer

class RobertaModel:
    DATASET: str = os.path.abspath(path="./data/mock/mock_dataset.csv")
    MESSAGE_COLUMN: str = "Message"

    # <--- Loading Important Values -->
    def __init__(self):
        tokeniser = RobertaTokenizer.from_pretrained("roberta-base")

        self.tokeniser = tokeniser

        tokeniser_loaded: str = "Tokeniser is loaded" if self.tokeniser  else "Tokeniser is not loaded"

        print(f"{"=" * 20}\n{tokeniser_loaded}\n{"=" * 20}\n")

    def data_loading(self):
        try:
            training_data: str = pd.read_csv(self.DATASET, on_bad_lines="skip")
            
            sample_messages: list = training_data[self.MESSAGE_COLUMN].head(3).tolist()

            if (len(sample_messages) < 1):
                raise KeyError(f"Error: The {self.MESSAGE_COLUMN} was not found.")
            else:
                print(f"\n<--- Found {len(sample_messages)} sample messages to tokenize: --->\n")

                for index, message in enumerate(sample_messages):
                    print(f"Sample {index + 1}: {message}")

                print("\n<--- Tokenising --->\n")

                inputs = self.tokeniser(
                    sample_messages,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )

                print("Tokenisation Complete!")

                print("\n<--- Output: Token Numbers --->\n")
                print(inputs["input_ids"])
                print(f"Shape: {inputs["input_ids"].shape}")

                print("\n<--- Output: attention_mask --->\n")
                print(inputs["attention_mask"])
                print(f"Shape: {inputs["attention_mask"].shape}")

                print("\n--- (Bonus) Decoding the first message to check ---")
                # Let's grab the IDs for the *first* message (index 0)
                first_message_ids = inputs['input_ids'][0]
                
                # .decode() converts the IDs back into text
                decoded_text = self.tokeniser.decode(first_message_ids)
                print(decoded_text)
                print("\nNotice the '<s>', '</s>', and '<pad>' tokens!")

        except FileNotFoundError:
            print(f"Error: The file was not found at path ({self.DATASET}).")
        except KeyError:
            print(f"Error: The {self.MESSAGE_COLUMN} was not found.")
        except Exception as e:
            print(f"An error occured: {e}")

roberta_model = RobertaModel()