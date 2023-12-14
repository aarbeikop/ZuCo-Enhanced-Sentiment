import torch
from scipy.stats import ttest_ind
from model import BertSentimentClassifier
from data import SentimentDataSet
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def load_model(path, hidden_size=400, num_labels=2, use_cognitive_features=True, cognitive_feature_size=5, dropout_prob=0.2):
    model = BertSentimentClassifier(hidden_size, num_labels, use_cognitive_features, cognitive_feature_size, dropout_prob)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)

    model.eval()
    return model

def predict(model, dataloader):
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            et_features = batch['et_features'] if 'et_features' in batch else None
            targets = batch['labels']

            outputs = model(inputs, attention_mask, et_features)
            preds = torch.round(outputs).squeeze().cpu().numpy()
            predictions.extend(preds)
            labels.extend(targets.cpu().numpy())
    
    return predictions, labels

def examine_specific_errors(dataset, labels, predictions):
    errors = []
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if pred != label:
            sentence = dataset.sentences_data.iloc[i]['sentence']
            errors.append((sentence, label, pred))
    return errors[:10]  

def print_error_analysis(labels, predictions):
    print("Classification Report:")
    print(classification_report(labels, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(labels, predictions))

def perform_t_test(data1, data2):
    # Assuming data1 and data2 are arrays of accuracies from multiple runs
    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)  # Use equal_var=False for unequal variances
    return t_stat, p_value

def plot_confusion_matrix(labels, predictions, file_name):
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(file_name)
    plt.close()

def main():
    model_path_dummy = 'best_model_dummy.pth'
    model_path_gaze = 'best_model_gaze.pth'
    dataset = SentimentDataSet('sentiment_labels_task1.csv', 'merged_word_data.csv')
    dataloader = dataset.get_split(range(len(dataset)))  # Load the full dataset

    # Evaluate the model with dummy features
    model_dummy = load_model(model_path_dummy, use_cognitive_features=False)
    predictions_dummy, labels_dummy = predict(model_dummy, dataloader)
    print("Error Analysis for Model with Dummy Features:")
    print_error_analysis(labels_dummy, predictions_dummy)
    errors_dummy = examine_specific_errors(dataset, labels_dummy, predictions_dummy)
    print("Sample Errors for Dummy Features Model:", errors_dummy)
    plot_confusion_matrix(labels_dummy, predictions_dummy, 'confusion_matrix-Dummy.png')

    # Evaluate the model with gaze features
    model_gaze = load_model(model_path_gaze, use_cognitive_features=True)
    predictions_gaze, labels_gaze = predict(model_gaze, dataloader)
    print("Error Analysis for Model with Gaze Features:")
    print_error_analysis(labels_gaze, predictions_gaze)
    errors_gaze = examine_specific_errors(dataset, labels_gaze, predictions_gaze)
    print("Sample Errors for Gaze Features Model:", errors_gaze)
    plot_confusion_matrix(labels_gaze, predictions_gaze, 'confusion_matrix-Gaze.png')

    # Perform t-test
    accuracy_dummy = accuracy_score(labels_dummy, predictions_dummy)
    accuracy_gaze = accuracy_score(labels_gaze, predictions_gaze)
    t_stat, p_value = perform_t_test([accuracy_dummy], [accuracy_gaze])
    print(f"T-test between models: T-statistic = {t_stat}, P-value = {p_value}")

if __name__ == '__main__':
    main()