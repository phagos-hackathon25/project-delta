import torch
import torch.nn as nn
import torch.nn.functional as F


hidden_dim = 1280
num_classes = 2
basic_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
class BaseEncoder(nn.Module):
    def __init__(self, client, output: str = "sequence"):
        """
        output: 'sequence' (default) or 'tokens'
        """
        super().__init__()
        self.client = client
        assert output in ["sequence", "tokens"], "output must be 'sequence' or 'tokens'"
        self.output = output

    def forward(self, protein_tensor):
        """
        protein_tensor: encoded tensor from client.encode(ESMProtein or batch)
        """
        logits_output = self.client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )

        # logits_output shape: [batch, seq_len, embedding_dim]
        if self.output == "tokens":
            return logits_output
        else:
            return logits_output.mean(dim=1)  # [batch, embedding_dim]

class BaseModel(nn.Module):
    def __init__(self, encoder: nn.Module, classifier: nn.Module, hidden_dim: int, num_classes: int):
        """
        encoder: pre-trained model (e.g., ESM, Evo, etc.)
        hidden_dim: dimension of output embedding from encoder
        num_classes: number of target classes
        """
        super(BaseModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x, attention_mask=None):
        """
        x: input sequences (tokenized)
        attention_mask: optional, for models like Evo
        """
        # Get embedding from encoder
        if attention_mask is not None:
            outputs = self.encoder(x, attention_mask=attention_mask)
        else:
            outputs = self.encoder(x)

        # Use mean pooling over tokens if needed
        if isinstance(outputs, tuple):
            hidden = outputs[0]  # (batch, seq_len, hidden)
        else:
            hidden = outputs  # (batch, hidden)

        if hidden.ndim == 3:
            hidden = hidden.mean(dim=1)  # average across sequence

        # Classification head
        logits = self.classifier(hidden)
        return logits

    def predict(self, x, attention_mask=None):
        logits = self.forward(x, attention_mask)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1), probs
