"""SCRIPT TO CHECK CONTENTS OF .MAT FILES"""
import scipy.io as io

# Define the path to your .mat file
file_path = "/Users/isabellecretton/Desktop/UGBERT/SEMESTER_3/COGN-EN-NLP/SEMINAR-PROJECT/ZuCo-Enhanced-Sentiment/sentencesSR.mat" # CHANGE THIS

# Load the .mat file
data = io.loadmat(file_path)

# Print the keys in the .mat file
print("Keys in the .mat file:")
for key in data.keys():
    print(key)

if 'sentences' in data:
    print("\nContents of 'sentencesSR':")
    print(data['sentences'])
else:
    print("\nKey 'sentencesSR' not found in the file.")