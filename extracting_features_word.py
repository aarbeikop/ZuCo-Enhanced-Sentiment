import os
import pandas as pd
import scipy.io as io
import numpy as np

def extract_word_data(file_path):
    data = io.loadmat(file_path, squeeze_me=True, struct_as_record=False)['sentenceData']
    words_data = []
    for sent in data:
        if hasattr(sent, 'word') and isinstance(sent.word, np.ndarray):
            for word in sent.word:
                word_info = {
                    'content': word.content,
                    'nFixations': getattr(word, 'nFixations', np.nan),
                    'FFD': getattr(word, 'FFD', np.nan),
                    'TRT': getattr(word, 'TRT', np.nan),
                    'GPT': getattr(word, 'GPT', np.nan),
                    'GD': getattr(word, 'GD', np.nan),
                }
                words_data.append(word_info)
    return words_data

folder_path = "/Users/isabellecretton/Desktop/UGBERT/SEMESTER_3/COGN-EN-NLP/SEMINAR-PROJECT/ZuCo-Enhanced-Sentiment/ZuCo/task1/Matlab files"

# Store all data and also for averaging
all_data = []
avg_data = []

for filename in os.listdir(folder_path):
    if filename.endswith('.mat'):
        file_path = os.path.join(folder_path, filename)
        word_data = extract_word_data(file_path)
        all_data.extend(word_data)
        avg_data.extend(word_data)  # Copy for averaging

# Convert to DataFrame
df_all = pd.DataFrame(all_data)
df_avg = pd.DataFrame(avg_data)

# Replace NaN with 0 for numerical columns in df_avg
numerical_cols = ['nFixations', 'FFD', 'TRT', 'GPT', 'GD']
df_avg[numerical_cols] = df_avg[numerical_cols].fillna(0)

# Compute the average for numerical columns
avg_df = df_avg.groupby('content', as_index=False)[numerical_cols].mean()

# Merge with the original DataFrame
merged_df = pd.merge(df_all, avg_df, on='content', suffixes=('', '_avg'))

# Drop duplicate content while keeping the first occurrence
final_df = merged_df.drop_duplicates(subset=['content'])

# Save the final DataFrame to a CSV file
final_df.to_csv('merged_word_data.csv', index=False)