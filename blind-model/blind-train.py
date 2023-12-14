import sys
import random
import numpy as np
from argparse import ArgumentParser
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.nn import BCELoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from blind import CognitiveFeatureClassifier
from blindata import CognitiveFeatureDataSet

def init_metrics():
    return {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}

def get_metrics(targets, predictions, scores, average_method):
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average=average_method, zero_division=0),
        'precision': precision_score(targets, predictions, average=average_method, zero_division=0),
        'recall': recall_score(targets, predictions, average=average_method, zero_division=0),
        'roc_auc': roc_auc_score(targets, scores[:, 1]) if average_method == 'binary' else roc_auc_score(targets, scores, multi_class='ovr')
    }
    return metrics

def print_metrics(metrics, prefix=''):
    print(f'{prefix} Accuracy: {metrics["accuracy"]:.4f}, Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}, ROC-AUC: {metrics["roc_auc"]:.4f}')

def print_mean_metrics(metrics, prefix=''):
    print(f'{prefix} Mean Accuracy: {np.mean(metrics["accuracy"]):.4f}, Std: {np.std(metrics["accuracy"]):.4f}')
    print(f'{prefix} Mean Precision: {np.mean(metrics["precision"]):.4f}, Std: {np.std(metrics["precision"]):.4f}')
    print(f'{prefix} Mean Recall: {np.mean(metrics["recall"]):.4f}, Std: {np.std(metrics["recall"]):.4f}')
    print(f'{prefix} Mean F1: {np.mean(metrics["f1"]):.4f}, Std: {np.std(metrics["f1"]):.4f}')
    print(f'{prefix} Mean ROC-AUC: {np.mean(metrics["roc_auc"]):.4f}, Std: {np.std(metrics["roc_auc"]):.4f}')

def iterate(dataloader, model, loss_fn, optimizer, l1_lambda=0.001, train=True):
    epoch_loss = 0.0
    all_targets = []
    all_predictions = []
    all_scores = []

    for batch in dataloader:
        et_features, targets = batch['cognitive_features'], batch['labels']
        if torch.cuda.is_available():
            et_features, targets = et_features.cuda(), targets.cuda()

        # Move this line outside of the CUDA conditional
        logits = model(et_features)
        targets = targets.float()
        loss = loss_fn(logits, targets)

        if l1_lambda > 0:
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_penalty

        if train:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss += loss.item()
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend((logits > 0.5).cpu().numpy())
        all_scores.extend(logits.cpu().detach().numpy())

    average_method = 'binary' if loss_fn == BCELoss else 'macro'
    return epoch_loss, all_scores, get_metrics(all_targets, all_predictions, all_scores, average_method)


def adjust_learning_rate(optimizer, factor=0.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor

def main():
    parser = ArgumentParser()
    parser.add_argument('--num-sentiments', type=int, default=2, help='2: binary classification, 3: ternary.')
    parser.add_argument('--use-gaze', action='store_true', help='Use gaze features if set')
    parser.add_argument('--word-features-file', type=str, help='Path to the word level features file')
    args = parser.parse_args()

    dataset = CognitiveFeatureDataSet('sentiment_labels_task1.csv', "merged_word_data.csv")
    lstm_units = 300
    loss_fn = BCELoss() 
    train_metrics = init_metrics()
    val_metrics = init_metrics()

    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_patience = 2
    adjustment_factor = 0.5  # Learning rate adjustment factor 

    labels = dataset.sentences_data['sentiment_label'].values
    print(sum([1 for l in labels if l == 0]))
    print(sum(labels))
    
   # train_val_index, test_index = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)
    #train_val_labels = labels[train_val_index]

    def analyze_errors(model, dataloader):
        model.eval()
        errors = []
        original_dataset = dataloader.dataset.dataset  # Accessing the original dataset
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                et_features, targets = batch['cognitive_features'], batch['labels']
                outputs = model(et_features)
                preds = torch.round(outputs).squeeze()
                indices = dataloader.dataset.indices  # Get the original indices of the samples in the subset
                for j, (pred, label) in enumerate(zip(preds, targets)):
                    if pred != label:
                        original_idx = indices[i * dataloader.batch_size + j]
                        sentence = original_dataset.sentences_data.iloc[original_idx]['sentence']
                        errors.append((sentence, label.item(), pred.item()))
        return errors

    
    # K-Fold Cross-Validation on the data
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    best_model_state = None
    best_val_loss = float('inf')
    train_metrics = init_metrics()
    val_metrics = init_metrics()
    test_metrics = init_metrics()

    for train_index, val_index in sss.split(np.zeros(len(labels)), labels):
        val_index, test_index = train_test_split(val_index, test_size=0.5, random_state=42)

        # Check for overlap of data (leakage)
        train_set = set(train_index)
        val_set = set(val_index)
        test_set = set(test_index)

        assert len(train_set & val_set) == 0, "Overlap found between training and validation sets"
        assert len(train_set & test_set) == 0, "Overlap found between training and test sets"
        assert len(val_set & test_set) == 0, "Overlap found between validation and test sets"

        train_loader = dataset.get_split(train_index)
        val_loader = dataset.get_split(val_index)
        test_loader = dataset.get_split(test_index)

        #print(f'sum of negative labels in train split {sum([1 for l in labels[train_index] if l == 0])}')
        #print(sum(labels[train_index]))

        model = model = CognitiveFeatureClassifier(lstm_units)
        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

        no_improvement_epochs = 0
        for e in range(10):
            # Adjusting the assignment to capture all three returned values
            train_loss, train_scores, train_results = iterate(train_loader, model, loss_fn, optimizer)
            val_loss, val_scores, val_results = iterate(val_loader, model, loss_fn, optimizer, train=False)

            print(f'\nEpoch {e + 1}:')
            print_metrics(train_results, 'TRAIN')
            print_metrics(val_results, 'VAL')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
                best_model_state = model.state_dict()
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs >= early_stopping_patience:
                print("Adjusting learning rate...")
                adjust_learning_rate(optimizer, factor=adjustment_factor)
                no_improvement_epochs = 0
                model.load_state_dict(best_model_state)

            scheduler.step(val_loss)

        # Accumulate metrics
        for metric in train_results:
            train_metrics[metric].append(train_results[metric])
            val_metrics[metric].append(val_results[metric])
        
        # Test Set Evaluation
        test_loss, test_scores, test_results = iterate(test_loader, model, loss_fn, optimizer, train=False)
        print('\nTest Metrics:')
        print_metrics(test_results, 'TEST')
        errors = analyze_errors(model, val_loader)
        print(f"Errors in Fold: {errors}")
        for metric in test_results:
            test_metrics[metric].append(test_results[metric])  

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, 'best_model.pth')
        model.load_state_dict(best_model_state)

    # Print mean metrics
    print('\n\n> 10-fold CV done')
    print_mean_metrics(train_metrics, 'TRAIN')
    print_mean_metrics(val_metrics, 'VAL')
    print_mean_metrics(test_metrics, 'TEST')
    
if __name__ == "__main__":
    main()