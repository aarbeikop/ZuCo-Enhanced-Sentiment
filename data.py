import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

class SentimentDataSet(Dataset):
    def __init__(self, sentences_file, word_features_file, max_length=128):
        self.sentences_data = pd.read_csv(sentences_file, delimiter=';')
        self.sentences_data = self.sentences_data.dropna(subset=['sentiment_label'])

        self.word_features = pd.read_csv(word_features_file)
        self.preprocess_features()  # Normalize the cognitive features

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def preprocess_features(self):
        # Replace empty lists with zeros and convert to floats
        for col in self.word_features.columns:
            if col != 'content':
                self.word_features[col] = self.word_features[col].apply(lambda x: 0.0 if x == '[]' else float(x))

        # Normalize the features using Z-score normalization
        scaler = StandardScaler()
        feature_cols = self.word_features.columns.drop('content')
        self.word_features[feature_cols] = scaler.fit_transform(self.word_features[feature_cols])

    def align_features_with_tokens(self, sentence, word_features):
        tokenized_sentence = self.tokenizer(sentence, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = tokenized_sentence['input_ids'].squeeze(0)

        aligned_features = []
        word_index = 0

        for token_id in input_ids:
            token = self.tokenizer.convert_ids_to_tokens(token_id.item())

            if token in ['[CLS]', '[SEP]']:
                aligned_features.append([0.0] * (word_features.shape[1] - 1))
            elif token.startswith('##') and word_index < len(word_features):
                aligned_features.append(word_features.iloc[word_index][1:].tolist())  # Repeat features for subword tokens
            else:
                if word_index < len(word_features) and token == word_features.iloc[word_index]['content']:
                    aligned_features.append(word_features.iloc[word_index][1:].tolist())
                    word_index += 1
                else:
                    aligned_features.append([0.0] * (word_features.shape[1] - 1))  # Default features for unmatched tokens

        return tokenized_sentence, torch.tensor(aligned_features, dtype=torch.float32)


    def __len__(self):
        return len(self.sentences_data)

    def __getitem__(self, idx):
        sentence_row = self.sentences_data.iloc[idx]
        sentence = sentence_row['sentence']

        # Initialize an empty DataFrame for word features
        word_features_for_sentence = pd.DataFrame(columns=self.word_features.columns)

        # Extract the words from the sentence
        words_in_sentence = sentence.split()

        # Fetch word features for each word in the sentence
        for word in words_in_sentence:
            if word in self.word_features['content'].values:
                word_feature = self.word_features[self.word_features['content'] == word].iloc[0]
                # Replace empty lists with zeros
                word_feature = word_feature.apply(lambda x: 0 if isinstance(x, list) else x)
                # Append the word features to the DataFrame
                word_features_for_sentence = pd.concat([word_features_for_sentence, pd.DataFrame([word_feature])], ignore_index=True)

        # Handle the case where no word features are found
        if word_features_for_sentence.empty:
            word_features_for_sentence = pd.DataFrame(0, index=np.arange(len(words_in_sentence)), columns=self.word_features.columns)

        # Align the features with the tokens, and return the tokenized sentence and the aligned features
        tokenized_sentence, et_features = self.align_features_with_tokens(sentence, word_features_for_sentence)

        # Get the original label and map it from [-1, 0, 1] to [0, 1, 2], due to the way CrossEntropyLoss expects the labels
        label = sentence_row['sentiment_label']
        label_mapped = int(label) + 1

        input_ids = tokenized_sentence['input_ids'].squeeze(0)
        attention_mask = tokenized_sentence['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'et_features': et_features,
            'labels': torch.tensor(label_mapped, dtype=torch.long)
        }


    def get_split(self, indices):
        return DataLoader(
            Subset(self, indices),
            batch_size=32, shuffle=True
        )

    def split_cross_val(self, num_folds=10):
        labels = self.sentences_data['sentiment_label'].values
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
            yield self.get_split(train_idx), self.get_split(test_idx)



