import os
import pandas as pd
import scipy.io as io
import numpy as np

def extract_sentence_data(file_path):
    data = io.loadmat(file_path, squeeze_me=True, struct_as_record=False)['sentenceData']
    sentences_data = []
    for sent in data:
        # Calculate the size of the wordbounds array
        wordbounds_size = len(sent.wordbounds.flatten()) if hasattr(sent, 'wordbounds') else 0

        sentence_info = {
            'content': sent.content,
            'omissionRate': getattr(sent, 'omissionRate', np.nan),
            'allFixations': getattr(sent, 'allFixations', np.nan),
            'wordboundsSize': wordbounds_size
        }
        sentences_data.append(sentence_info)
    return sentences_data

folder_path = "/Users/isabellecretton/Desktop/UGBERT/SEMESTER_3/COGN-EN-NLP/SEMINAR-PROJECT/ZuCo-Enhanced-Sentiment/ZuCo/task1/Matlab files"

all_data = []
for filename in os.listdir(folder_path):
    if filename.endswith('.mat'):
        file_path = os.path.join(folder_path, filename)
        sentence_data = extract_sentence_data(file_path)
        all_data.extend(sentence_data)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Replace NaN with 0 for numerical columns
numerical_cols = ['omissionRate', 'allFixations', 'wordboundsSize']
df[numerical_cols] = df[numerical_cols].fillna(0)

# Compute the average for numerical columns
avg_df = df.groupby('content', as_index=False)[numerical_cols].mean()

# Save the average data to a CSV file
avg_df.to_csv('average_sentence_data.csv', index=False)