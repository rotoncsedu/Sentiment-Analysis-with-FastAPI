
import json
import re

import torch

from model import GRUModel

ARTIFACTS_DIR = "artifacts"

def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)    # keep only letters
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    return text

def encode(text: str, word2idx: dict, max_len: int) -> list:
    tokens = text.split()
    ids = [word2idx.get(t, 1) for t in tokens]  # 1 = <UNK>
    ids = ids[:max_len]                          # truncate
    ids = ids + [0] * (max_len - len(ids))       # pad
    return ids


class SentimentPredictor:
    """
    Loads config, vocabulary, and model weights once at startup,
    then serves predictions via .predict(text).
    """

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        # 1. Load config.json -> tells us the exact architecture shape
        with open(f"{artifacts_dir}/config.json") as f:
            self.config = json.load(f)

        # 2. Load word2idx.json -> tokenization
        with open(f"{artifacts_dir}/word2idx.json") as f:
            self.word2idx = json.load(f)

        self.max_len = self.config["max_len"]

        # 3. Reconstruct GRUModel with the saved hyperparameters
        self.model = GRUModel(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
        )

        # 4. Load weights. 
        state_dict = torch.load(
            f"{artifacts_dir}/gru_model.pt", map_location="cpu"
        )
        self.model.load_state_dict(state_dict)

        # Disables dropout during inference
        self.model.eval()

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        ids = encode(cleaned, self.word2idx, self.max_len)
        x = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()

        label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
        return {"label": label, "probability": prob}