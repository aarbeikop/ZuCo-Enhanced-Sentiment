import torch
from model import BertSentimentClassifier
from data import SentimentDataSet
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def load_model(path, hidden_size=400, num_labels=2, use_cognitive_features=False, cognitive_feature_size=5, dropout_prob=0.2):
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

def plot_confusion_matrix(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig('confusion_matrix-Enhanced.png')

def main():
    model_path = 'best_model.pth'
    dataset = SentimentDataSet('sentiment_labels_task1.csv', 'merged_word_data.csv')
    dataloader = dataset.get_split(range(len(dataset)))  # Load the full dataset

    model = load_model(model_path)
    predictions, labels = predict(model, dataloader)

    print_error_analysis(labels, predictions)
    errors = examine_specific_errors(dataset, labels, predictions)
    print("Sample Errors:", errors)

    plot_confusion_matrix(labels, predictions)

    # Further analysis can be added here, such as:
    # - Inspecting specific examples of errors
    # - Correlating errors with specific features or data properties
    # - Advanced statistical analysis or visualizations

if __name__ == '__main__':
    main()