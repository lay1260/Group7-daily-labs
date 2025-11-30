import torch
import torch.nn as nn


class FCModel(nn.Module):
    """
    Fully Connected Model for binary classification.
    Takes BERT pooled output (768-dim) and outputs probability score.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        # Two-layer fully connected network
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, bert_pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            bert_pooled_output: BERT pooled output tensor of shape (batch_size, 768)
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        hidden = self.linear1(bert_pooled_output)
        hidden = self.activation(hidden)
        output = self.linear2(hidden)
        output = self.output_activation(output)
        return output
