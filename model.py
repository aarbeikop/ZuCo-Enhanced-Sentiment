import torch
from torch import nn
from transformers import BertModel

# used model.py from https://github.com/chipbautista/zuco-sentiment-analysis to get me started

class BertSentimentClassifier(nn.Module):
    bert_output_size = None  # Define bert_output_size as a class attribute

    def __init__(self, hidden_size, num_labels, cognitive_feature_size=5, dropout_prob=0.2):
        super(BertSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.cognitive_feature_size = cognitive_feature_size
        self.num_labels = num_labels

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Define weights for layer combination
        self.layer_weights = nn.Parameter(torch.ones(self.bert.config.num_hidden_layers))

        # Calculate bert_output_size based on the BERT model configuration
        BertSentimentClassifier.bert_output_size = self.bert.config.hidden_size 
        #print("bert_output_size:", BertSentimentClassifier.bert_output_size)  # Print for verification
        lstm_input_size = BertSentimentClassifier.bert_output_size + 5 # had to hardcode this, it was not working otherwise even though the default parameter is 5

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(2 * hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Attention layer (kept from original code, only using one)
        self.attention = nn.Linear(2 * hidden_size, 1) 

        # Final classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, cognitive_features=None):
        # Process input through BERT with no gradient updates
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        # Calculate a weighted sum of the BERT hidden states
        weighted_hidden_states = sum(w * h for w, h in zip(self.layer_weights, hidden_states))
        combined_hidden_states = weighted_hidden_states / torch.sum(self.layer_weights)

        # Concatenate cognitive features to the BERT output if they are provided

        combined_input = torch.cat((combined_hidden_states, cognitive_features), dim=-1)

        # Process the combined input through the LSTM layer and normalize it
        lstm_output, _ = self.lstm(combined_input) # output is (batch_size, seq_len, 2 * hidden_size)
        lstm_output = self.layer_norm(lstm_output) 

        # Apply attention to the LSTM output
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        # Apply dropout to the context vector and classify using the final layer
        context_vector = self.dropout(context_vector)
        logits = self.classifier(context_vector)

        return logits
