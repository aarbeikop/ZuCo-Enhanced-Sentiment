import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

class SentimentDataSet(Dataset):
    def __init__(self, sentences_file, word_features_file, max_length=500, use_dummy_features=False):
        self.sentences_data = pd.read_csv(sentences_file, delimiter=';')
        self.sentences_data = self.sentences_data.dropna(subset=['sentiment_label'])
        self.use_dummy_features = use_dummy_features

        # Convert sentiment labels to binary (0: non-positive, 1: positive) 
        self.sentences_data['sentiment_label'] = self.sentences_data['sentiment_label'].apply(lambda x: 0 if x == -1 else 1)

        self.word_features = pd.read_csv(word_features_file)
        self.preprocess_features()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = self.get_max_token_length(self.sentences_data['sentence'])

    def get_max_token_length(self, sentences):
        max_len = 0
        for sentence in sentences:
            # Tokenize the sentence and get the number of tokens
            input_ids = self.tokenizer.encode(sentence, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        return max_len

    def preprocess_features(self):
        # Replace empty lists with zeros and convert to floats. 
        for col in self.word_features.columns:
            if col != 'content':
                self.word_features[col] = self.word_features[col].apply(lambda x: 0.0 if x == '[]' else float(x))

        # Normalize the features using Z-score normalization
        #scaler = StandardScaler()
        #feature_cols = self.word_features.columns.drop('content')
        #self.word_features[feature_cols] = scaler.fit_transform(self.word_features[feature_cols])

    def align_features_with_tokens(self, sentence, word_features):
        tokenized_sentence = self.tokenizer(sentence, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = tokenized_sentence['input_ids'].squeeze(0)

        aligned_features = []
        word_index = 0

        for i, token_id in enumerate(input_ids):
            token = self.tokenizer.convert_ids_to_tokens(token_id.item())

            if token in ['[CLS]', '[SEP]', '[PAD]']:
                aligned_features.append([0.0] * 5)  # each word has 5 features, nFixations, FFD, GPT, TRT, 
            elif token.startswith('##') and word_index < len(word_features):
                # use features of the word that this subword is part of
                aligned_features.append(word_features.iloc[word_index][1:6].tolist())
            else:
                # find the word feature index for the current token
                while word_index < len(word_features) and token != word_features.iloc[word_index]['content'] and not token.startswith('##'):
                    word_index += 1

                if word_index < len(word_features):
                    aligned_features.append(word_features.iloc[word_index][6:].tolist()) # we only want the averaged features
                else:
                    aligned_features.append([0.0] * 5)  # default to zero if no matching feature found

                if not token.startswith('##'):
                    word_index += 1

        return tokenized_sentence, torch.tensor(aligned_features, dtype=torch.float32)



    def __len__(self):
        return len(self.sentences_data)

    def __getitem__(self, idx):
        sentence_row = self.sentences_data.iloc[idx]
        sentence = sentence_row['sentence']

        # initialize an empty DataFrame for word features
        word_features_for_sentence = pd.DataFrame(columns=self.word_features.columns)

        # extract the words from the sentence
        words_in_sentence = sentence.split()

        # get word features for each word in the sentence
        for word in words_in_sentence:
            if word in self.word_features['content'].values:
                word_feature = self.word_features[self.word_features['content'] == word].iloc[0]
                # Replace empty lists with zeros and convert to floats.
                word_feature = word_feature.apply(lambda x: 0 if isinstance(x, list) else x)
                # Append the word features to the df
                word_features_for_sentence = pd.concat([word_features_for_sentence, pd.DataFrame([word_feature])], ignore_index=True)

        # handle the case where no word features are found
        if word_features_for_sentence.empty:
            word_features_for_sentence = pd.DataFrame(0, index=np.arange(len(words_in_sentence)), columns=self.word_features.columns)

        # align the features with the tokens, and return the tokenized sentence and the aligned features
        tokenized_sentence, et_features = self.align_features_with_tokens(sentence, word_features_for_sentence)
        

        # convert the sentiment label from [-1, 0, 1] to binary [0, 1]
        label = sentence_row['sentiment_label']
        label_binary = 0 if label == 0 else 1

        input_ids = tokenized_sentence['input_ids'].squeeze(0)
        attention_mask = tokenized_sentence['attention_mask'].squeeze(0)

        tokenized_sentence, real_et_features = self.align_features_with_tokens(sentence, word_features_for_sentence)

        if self.use_dummy_features:
            # Generate dummy cognitive features for this sentence
            et_features = self.get_dummy_features(len(tokenized_sentence['input_ids'].squeeze(0)))
        else:
            et_features = real_et_features

        label_binary = 0 if sentence_row['sentiment_label'] == 0 else 1

        return {
            'input_ids': tokenized_sentence['input_ids'].squeeze(0),
            'attention_mask': tokenized_sentence['attention_mask'].squeeze(0),
            'et_features': et_features,
            'labels': torch.tensor(label_binary, dtype=torch.long)
        }

    def get_dummy_features(self, num_tokens):
        # Generate dummy features for each token in the sentence
        dummy_features = []
        for _ in range(num_tokens):
            single_feature = []
            for v in self.word_features.columns[1:]:
                min_val = self.word_features[v].min()
                max_val = self.word_features[v].max()
                single_feature.append(np.random.uniform(min_val, max_val))
            dummy_features.append(single_feature)

        return torch.tensor(dummy_features, dtype=torch.float32)

    def get_split(self, indices):
        return DataLoader(
            Subset(self, indices),
            batch_size=16, shuffle=True
        )

    def split_cross_val(self, num_folds=10):
        labels = self.sentences_data['sentiment_label'].values
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
            yield self.get_split(train_idx), self.get_split(test_idx)




