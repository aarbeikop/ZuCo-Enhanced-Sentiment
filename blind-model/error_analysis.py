import torch
from sklearn.metrics import classification_report, confusion_matrix
from blind import CognitiveFeatureClassifier
from blindata import CognitiveFeatureDataSet
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load the model
def load_model(path, hidden_size=300):
    model = CognitiveFeatureClassifier(hidden_size)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to make predictions using the model
def predict(model, dataloader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            et_features = batch['cognitive_features']
            targets = batch['labels']
            outputs = model(et_features)
            preds = torch.round(outputs).squeeze().cpu().numpy()
            predictions.extend(preds)
            labels.extend(targets.cpu().numpy())
    return predictions, labels

def plot_confusion_matrix(labels, predictions, title='Confusion Matrix'):
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def examine_errors(dataset, predictions, labels):
    errors = pd.DataFrame(columns=['Sentence', 'True Label', 'Predicted Label'])
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if pred != label:
            sentence = dataset.sentences_data.iloc[i]['sentence']
            errors = errors.append({'Sentence': sentence, 'True Label': label, 'Predicted Label': pred}, ignore_index=True)
    return errors.head(10)  # Display the first 10 errors

def main():
    model_path = 'best_model.pth'
    model = load_model(model_path, hidden_size=300)
    dataset = CognitiveFeatureDataSet('sentiment_labels_task1.csv', 'merged_word_data.csv')
    test_loader = dataset.get_split(range(len(dataset)))

    predictions, labels = predict(model, test_loader)
    
    print("Classification Report:")
    print(classification_report(labels, predictions))

    plot_confusion_matrix(labels, predictions)

    errors = examine_errors(test_loader.dataset, predictions, labels)
    print("Sample Errors:\n", errors)

if __name__ == '__main__':
    main()
