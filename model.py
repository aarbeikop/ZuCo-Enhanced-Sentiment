import torch
from torch import nn
from transformers import BertModel

class BertSentimentClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels, use_cognitive_features=False, cognitive_feature_size=5, dropout_prob=0.1, fine_tune_layers=4):
        super(BertSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.use_cognitive_features = use_cognitive_features
        self.cognitive_feature_size = cognitive_feature_size
        self.num_labels = num_labels
        for param in self.bert.parameters():
            param.requires_grad = False

        bert_output_size = self.bert.config.hidden_size
        if use_cognitive_features:
            bert_output_size += cognitive_feature_size

        self.lstm = nn.LSTM(input_size=bert_output_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(2 * hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Attention layer
        self.attention = nn.Linear(2 * hidden_size, 1)

        # Final classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, 1), # Output is now a single value because I changed from ternary to binary classification and we use BCELoss
            nn.Sigmoid()  # Sigmoid activation for binary classification, couldve used nn.BCEWithLogitsLoss() instead of nn.BCELoss()
        )

    def forward(self, input_ids, attention_mask, cognitive_features=None):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'] 

        if self.use_cognitive_features and cognitive_features is not None:
            combined_input = torch.cat((bert_output, cognitive_features), dim=2)
        else:
            combined_input = bert_output

        lstm_output, _ = self.lstm(combined_input)
        lstm_output = self.layer_norm(lstm_output)

        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        context_vector = self.dropout(context_vector)
        logits = self.classifier(context_vector)

        return logits
