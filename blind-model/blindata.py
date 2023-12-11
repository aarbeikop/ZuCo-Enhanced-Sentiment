import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler

class CognitiveFeatureDataSet(Dataset):
    def __init__(self, sentences_file, cognitive_features_file, sequence_length=180):
        self.sentences_data = pd.read_csv(sentences_file, delimiter=';')
        self.sentences_data = self.sentences_data.dropna(subset=['sentiment_label'])
        self.sentences_data['sentiment_label'] = self.sentences_data['sentiment_label'].apply(lambda x: 0 if x == -1 else 1)
        self.word_features = pd.read_csv(cognitive_features_file)
        self.preprocess_features()
        self.max_feature_length = self.get_max_feature_length()
        self.sequence_length = sequence_length

    def preprocess_features(self):
        # Replace empty lists with zeros and convert to floats
        for col in self.word_features.columns:
            if col != 'content':
                self.word_features[col] = self.word_features[col].apply(lambda x: 0.0 if x == '[]' else float(x))

        # Normalize the features using Z-score normalization
        scaler = StandardScaler()
        feature_cols = self.word_features.columns.drop('content')
        self.word_features[feature_cols] = scaler.fit_transform(self.word_features[feature_cols])

    def get_max_feature_length(self):
        max_length = 0
        for sentence in self.sentences_data['sentence']:
            words = sentence.split()
            length = sum(len(self.word_features[self.word_features['content'] == word].iloc[0][1:].tolist()) for word in words if word in self.word_features['content'].values)
            max_length = max(max_length, length)
        return max_length

    def __len__(self):
        return len(self.sentences_data)

    def __getitem__(self, idx):
        sentence_row = self.sentences_data.iloc[idx]
        label = sentence_row['sentiment_label']

        # Fetch cognitive features for each word in the sentence
        sentence = sentence_row['sentence']
        words = sentence.split()
        cognitive_features = []
        for word in words:
            if word in self.word_features['content'].values:
                features = self.word_features[self.word_features['content'] == word].iloc[0][1:].tolist()
                cognitive_features.extend(features)
            else:
                cognitive_features.extend([0.0] * self.sequence_length)

        # Pad or truncate the feature vector to a fixed length
        fixed_length = 200  # Set your desired fixed length
        if len(cognitive_features) > fixed_length:
            cognitive_features = cognitive_features[:fixed_length]
        else:
            cognitive_features.extend([0.0] * (fixed_length - len(cognitive_features)))



        return {
            'cognitive_features': torch.tensor(cognitive_features, dtype=torch.float).view(1, fixed_length),  # Adjust view for LSTM input
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def get_split(self, indices):
        return DataLoader(
            Subset(self, indices),
            batch_size=16, shuffle=True
        )

if __name__ == "__main__":
    dataset = CognitiveFeatureDataSet('sentiment_labels_task1.csv', 'merged_word_data.csv')
    for data in dataset:
        features, label = data['cognitive_features'], data['labels']
        print(features.size(), label)
