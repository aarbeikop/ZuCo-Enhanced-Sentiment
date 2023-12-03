import torch
from torch import nn
from transformers import BertModel

class BertSentimentClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels, use_cognitive_features=True
                 , cognitive_feature_size=5):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = num_labels
        self.use_cognitive_features = use_cognitive_features
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        bert_output_size = self.bert.config.hidden_size
        if use_cognitive_features:
            bert_output_size += cognitive_feature_size

        # Using a bidirectional LSTM
        self.lstm = nn.LSTM(input_size=bert_output_size, 
                            hidden_size=hidden_size, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True)

        # Attention layer
        self.attention = nn.Linear(2 * hidden_size, 1)

        # Adjusting the classifier to take the concatenated output
        self.classifier = nn.Linear(2 * hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, cognitive_features=None):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']

        if self.use_cognitive_features and cognitive_features is not None:
            combined_input = torch.cat((bert_output, cognitive_features), dim=2)
        else:
            combined_input = bert_output

        lstm_output, _ = self.lstm(combined_input)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        logits = self.classifier(context_vector)
        return logits