# ZuCo-Enhanced-Sentiment

# Sentiment Analysis with BERT, LSTM, and Cognitive Features

## Overview
This project focuses on sentiment analysis using the ZuCo dataset. It involves training two classifiers: one enhanced with cognitive features and the other without. The objective is to determine if the inclusion of cognitive features, such as reading measures, improves the classification accuracy for sentiment analysis.

This work is inspired by the research presented in the paper "Advancing NLP with Cognitive Language Processing Signals" (https://arxiv.org/pdf/1904.02682.pdf) and utilizes the ZuCo dataset for exploring cognitive feature enhancement in sentiment analysis.

Code from https://github.com/chipbautista/zuco-sentiment-analysis was used.

## Features
- Utilizes BERT for word-level representations.
- Incorporates LSTM for sequential modeling and sentence-level representation.
- Employs a binary classification task (positive/negative sentiments).
- Includes two classifiers: one with cognitive feature enhancement and one without. The classifiers are defined within the parameters, where use_cognitive_features is either set to True or False
- Freezes transformer parameters, training only the sequence model and linear layer parameters.
- Compares performance to assess the impact of cognitive feature enhancement.

## Requirements
- Python 3.8 or above
- PyTorch
- Transformers
- Sklearn
- Pandas
- Seaborn and Matplotlib for visualization

## Dataset
The project uses the ZuCo dataset for sentiment analysis. You can find the dataset here: https://osf.io/q3zws/
or here: https://github.com/DS3Lab/zuco-nlp/tree/master/sentiment-analysis

## Model Architecture
The `BertSentimentClassifier` in `model.py` defines two model architectures:
1. Standard BERT-LSTM for sentiment classification.
2. BERT-LSTM with cognitive feature enhancement (Eye-Tracking).

## Usage
To train the models, run the `train.py` script:
with gaze features:
python train.py  --use-gaze --word-features-file path_to_word_features_file.csv
with dummy features:
python train.py --word-features-file path_to_word_features_file.csv




