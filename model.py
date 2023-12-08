import torch
from torch import nn
from transformers import BertModel

class BertSentimentClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels, use_cognitive_features=True, cognitive_feature_size=5, dropout_prob=0.2):
        super(BertSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.use_cognitive_features = use_cognitive_features
        self.cognitive_feature_size = cognitive_feature_size
        self.num_labels = num_labels

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Define weights for layer combination
        self.layer_weights = nn.Parameter(torch.ones(self.bert.config.num_hidden_layers))

        bert_output_size = self.bert.config.hidden_size
        lstm_input_size = bert_output_size + cognitive_feature_size if use_cognitive_features else bert_output_size

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(2 * hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Attention layer
        self.attention = nn.Linear(2 * hidden_size, 1) 

        # Final classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, cognitive_features=None):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        # Weighted sum of hidden states
        weighted_hidden_states = sum(w * h for w, h in zip(self.layer_weights, hidden_states)) # Element-wise multiplication of weights and hidden states 
        combined_hidden_states = weighted_hidden_states / torch.sum(self.layer_weights) # Normalize weights to sum to 1

        # If using cognitive features, concatenate them 
        if self.use_cognitive_features and cognitive_features is not None:
            combined_input = torch.cat((combined_hidden_states, cognitive_features), dim=-1)
        else:
            combined_input = combined_hidden_states

        # Process combined input through the LSTM
        lstm_output, _ = self.lstm(combined_input)
        lstm_output = self.layer_norm(lstm_output)

        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        context_vector = self.dropout(context_vector)
        logits = self.classifier(context_vector)

        return logits
