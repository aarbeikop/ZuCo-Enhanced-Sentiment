import torch
from torch import nn

class CognitiveFeatureClassifier(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.2, sequence_length=200):
        super(CognitiveFeatureClassifier, self).__init__()
        self.sequence_length = sequence_length
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=sequence_length, 
                            hidden_size=hidden_size, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(2 * hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Attention layer
        self.attention = nn.Linear(2 * hidden_size, 1)

        # Final classifier layer - Outputting a single value for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, 1),  # Outputting a single value
            nn.Sigmoid()
        )

    def forward(self, cognitive_features):
        # Process cognitive features through the LSTM
        lstm_output, _ = self.lstm(cognitive_features)
        lstm_output = self.layer_norm(lstm_output)

        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        context_vector = self.dropout(context_vector)
        logits = self.classifier(context_vector)
        logits = logits.squeeze()  # Ensure logits are of shape [batch_size]

        return logits
