import sys
import random
import numpy as np
from argparse import ArgumentParser
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.nn import BCELoss, CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from model import BertSentimentClassifier
from data import SentimentDataSet


def init_metrics():
    return {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}

def get_metrics(targets, predictions, scores, average_method):
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average=average_method, zero_division=0),
        'precision': precision_score(targets, predictions, average=average_method, zero_division=0),
        'recall': recall_score(targets, predictions, average=average_method, zero_division=0),
    }
    if average_method == 'binary':
        metrics['roc_auc'] = roc_auc_score(targets, scores[:, 1])
    elif average_method == 'macro':
        metrics['roc_auc'] = roc_auc_score(targets, scores, multi_class='ovr')
    return metrics

def print_metrics(metrics, prefix=''):
    print(f'{prefix} Accuracy: {metrics["accuracy"]:.4f}, Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}, ROC-AUC: {metrics["roc_auc"]:.4f}')

def print_mean_metrics(metrics, prefix=''):
    print(f'{prefix} Accuracy: {np.mean(metrics["accuracy"]):.4f}, Precision: {np.mean(metrics["precision"]):.4f}, Recall: {np.mean(metrics["recall"]):.4f}, F1: {np.mean(metrics["f1"]):.4f}, ROC-AUC: {np.mean(metrics["roc_auc"]):.4f}')

def iterate(dataloader, model, loss_fn, optimizer, l1_lambda=0.001, train=True):
    epoch_loss = 0.0
    all_targets = []
    all_predictions = []
    all_scores = []

    for batch in dataloader:
        input_ids, attention_mask, et_features, targets = batch['input_ids'], batch['attention_mask'], batch['et_features'], batch['labels']
        if torch.cuda.is_available():
            input_ids, attention_mask, targets = input_ids.cuda(), attention_mask.cuda(), targets.cuda()
            et_features = et_features.cuda() if et_features is not None else None

        with torch.autograd.set_detect_anomaly(True):
            logits = model(input_ids, attention_mask, et_features)

            logits = logits.squeeze()
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

        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend((logits > 0.5).cpu().numpy())
        all_scores.extend(logits.cpu().detach().numpy())

    average_method = 'binary' if loss_fn == BCELoss else 'macro'
    return epoch_loss, all_scores, get_metrics(all_targets, all_predictions, all_scores, average_method)

def main():
    parser = ArgumentParser()
    parser.add_argument('--num-sentiments', type=int, default=2, help='2: binary classification, 3: ternary.')
    parser.add_argument('--use-gaze', action='store_true', help='Use gaze features if set')
    parser.add_argument('--word-features-file', type=str, required=True, help='Path to the word level features file')
    args = parser.parse_args()

    dataset = SentimentDataSet('sentiment_labels_task1.csv', args.word_features_file)
    lstm_units = 400
    loss_fn = BCELoss() 
    train_metrics = init_metrics()
    val_metrics = init_metrics()

    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_patience = 5

    labels = dataset.sentences_data['sentiment_label'].values
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    for train_index, test_index in sss.split(np.zeros(len(labels)), labels):
        train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=42)
        train_loader = dataset.get_split(train_index)
        val_loader = dataset.get_split(val_index)
        test_loader = dataset.get_split(test_index)

        model = BertSentimentClassifier(lstm_units, args.num_sentiments, args.use_gaze)
        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=2e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

        no_improvement_epochs = 0
        for e in range(10):
            train_loss, train_scores, train_results = iterate(train_loader, model, loss_fn, optimizer)
            val_loss, val_scores, val_results = iterate(val_loader, model, loss_fn, optimizer, train=False)
            test_loss, test_scores, test_results = iterate(test_loader, model, loss_fn, optimizer, train=False)

            print(f'\nEpoch {e + 1}:')
            print_metrics(train_results, 'TRAIN')
            print_metrics(val_results, 'VAL')
            print_metrics(test_results, 'TEST')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
                best_model_state = model.state_dict()
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs >= early_stopping_patience:
                print("Early stopping triggered")
                break

            scheduler.step(val_loss)

        # Accumulate metrics
        for metric in train_results:
            train_metrics[metric].append(train_results[metric])
            val_metrics[metric].append(val_results[metric])
            #test_results[metric].append(test_results[metric]) can't append numpy.float

    if best_model_state is not None:
        torch.save(best_model_state, 'best_model.pth')

    # Print mean metrics
    print('\n\n> 5-fold CV done')
    print_mean_metrics(train_metrics, 'TRAIN')
    print_mean_metrics(val_metrics, 'VAL')
    #print_mean_metrics(test_results, 'TEST')

if __name__ == "__main__":
    main()