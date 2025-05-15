import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from esm.sdk.api import ESMProtein, LogitsConfig
from typing import Dict

hidden_dim = 960
num_classes = 2
basic_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

class ProteinDataset(Dataset):
    def __init__(self, protein_list, labels):
        self.protein_list = protein_list
        self.labels = labels

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        return self.protein_list[idx], self.labels[idx]

class ESMEncoder(nn.Module):
    def __init__(self, client, output: str = "sequence"):
        """
        output: 'sequence' (default) or 'tokens'
        """
        super().__init__()
        self.client = client
        assert output in ["sequence", "tokens"], "output must be 'sequence' or 'tokens'"
        self.output = output

    def forward(self, protein: ESMProtein):
        """
        protein_tensor: encoded tensor from client.encode(ESMProtein or batch)
        """
        protein_tensor = self.client.encode(protein)
        logits_output = self.client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        ).embeddings

        # logits_output shape: [batch, seq_len, embedding_dim]
        if self.output == "tokens":
            return logits_output
        else:
            return logits_output.mean(dim=1)  # [batch, embedding_dim]

import os
import torch
import hashlib

class BaseModel(nn.Module):
    def __init__(self, encoder: nn.Module, classifier: nn.Module, cache_dir: str = "./embedding_cache"):
        super(BaseModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self._cache: Dict[str, torch.Tensor] = {}
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, protein_id: str) -> str:
        # Use SHA1 to avoid filesystem issues
        filename = hashlib.sha1(protein_id.encode()).hexdigest() + ".pt"
        return os.path.join(self.cache_dir, filename)

    def encode_with_cache(self, protein):
        protein_id = getattr(protein, "id", None) or protein.sequence
        cache_path = self._get_cache_path(protein_id)

        if protein_id in self._cache:
            return self._cache[protein_id]

        if os.path.exists(cache_path):
            embedding = torch.load(cache_path)
            embedding = embedding.detach().to(self.classifier[0].weight.device)
            self._cache[protein_id] = embedding
            return embedding

        with torch.no_grad():
            embedding = self.encoder(protein).detach()
            torch.save(embedding, cache_path)
            self._cache[protein_id] = embedding
            return embedding

    def forward(self, x, attention_mask=None):
        if isinstance(x, list):
            hidden = torch.stack([self.encode_with_cache(p) for p in x])
        else:
            hidden = self.encode_with_cache(x)

        if hidden.ndim == 3:
            hidden = hidden.mean(dim=1)  # mean pooling over tokens

        return self.classifier(hidden)

    def predict(self, x, attention_mask=None):
        logits = self.forward(x, attention_mask)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return torch.argmax(probs, dim=-1), probs, entropy
