import sys
import csv
import os
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
from sklearn.preprocessing import StandardScaler

from model import BertSentimentClassifier
from data import SentimentDataSet

# used train.py from https://github.com/chipbautista/zuco-sentiment-analysis to get me started

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_metrics():
    return {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}

def get_metrics(targets, predictions, scores, average_method):
    scores_array = np.array(scores)
    # Now, you can use scores_array in roc_auc_score
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average=average_method, zero_division=0),
        'precision': precision_score(targets, predictions, average=average_method, zero_division=0),
        'recall': recall_score(targets, predictions, average=average_method, zero_division=0),
        'roc_auc': roc_auc_score(targets, scores_array) if average_method == 'binary' else roc_auc_score(targets, scores_array, multi_class='ovr')
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
        input_ids, attention_mask, et_features, targets = batch['input_ids'], batch['attention_mask'], batch['et_features'], batch['labels']
        if torch.cuda.is_available():
            input_ids, attention_mask, targets = input_ids.cuda(), attention_mask.cuda(), targets.cuda()
            et_features = et_features.cuda() if et_features is not None else None

        with torch.autograd.set_detect_anomaly(True): # For debugging
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
                clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping to prevent exploding gradients 
                optimizer.step()

        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend((logits > 0.5).cpu().numpy())
        all_scores.extend(logits.cpu().detach().numpy())

    average_method = 'binary' if len(set(all_targets)) == 2 else 'macro'
    return epoch_loss, all_scores, get_metrics(all_targets, all_predictions, all_scores, average_method)

def adjust_learning_rate(optimizer, factor=0.5): 
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor

def create_folder_if_not_exists(folder_path):
    """Create a folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

def write_to_csv(file_path, data, headers=None, mode='a'):
    """Write data to a CSV file."""
    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)
        if headers and mode == 'w':
            writer.writerow(headers)
        writer.writerow(data)


def main():
    set_global_seed(42)
    parser = ArgumentParser()
    parser.add_argument('--num-sentiments', type=int, default=2, help='2: binary classification, 3: ternary.')
    parser.add_argument('--use-gaze', action='store_true', help='Use gaze features if set')
    parser.add_argument('--word-features-file', type=str, required=True, help='Path to the word level features file')
    args = parser.parse_args()

    if args.use_gaze == True:
        dataset = SentimentDataSet('sentiment_labels_task1.csv', args.word_features_file, use_dummy_features=False)
    else:
        dataset = SentimentDataSet('sentiment_labels_task1.csv', args.word_features_file, use_dummy_features=True)
    
    lstm_units = 400
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
    
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    best_model_state = None
    best_val_loss = float('inf')
    test_metrics = init_metrics()

    base_metrics_folder = os.path.join(os.getcwd(), 'model_metrics')
    create_folder_if_not_exists(base_metrics_folder)

    fold_number = 0
    all_test_metrics = []
    for fold_idx, (train_index, val_index) in enumerate(sss.split(np.zeros(len(labels)), labels), 1):
        fold_number += 1
        fold_metrics_folder = os.path.join(base_metrics_folder, f'fold_{fold_number}')
        create_folder_if_not_exists(fold_metrics_folder)

        # Split the validation set into validation and test sets
        val_index, test_index = train_test_split(val_index, test_size=0.5, random_state=42)
        train_loader = dataset.get_split(train_index)
        val_loader = dataset.get_split(val_index)
        test_loader = dataset.get_split(test_index)

        # Create a subfolder for each fold
        if not os.path.exists(fold_metrics_folder):
            os.makedirs(fold_metrics_folder)
            print(f"Created folder for fold {fold_number}")
        
        
        scaler = StandardScaler()
        train_features = dataset.word_features.iloc[train_index]
        val_features = dataset.word_features.iloc[val_index]
        test_features = dataset.word_features.iloc[test_index]

        feature_cols = [col for col in train_features.columns if col != 'content']
        train_features[feature_cols] = scaler.fit_transform(train_features[feature_cols])
        val_features[feature_cols] = scaler.transform(val_features[feature_cols])
        test_features[feature_cols] = scaler.transform(test_features[feature_cols])

        dataset.word_features.iloc[train_index] = train_features
        dataset.word_features.iloc[val_index] = val_features
        dataset.word_features.iloc[test_index] = test_features

        model = BertSentimentClassifier(lstm_units, args.num_sentiments)
        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

        no_improvement_epochs = 0
        for e in range(10):
            _, _, train_results = iterate(train_loader, model, loss_fn, optimizer)
            val_loss, _, val_results = iterate(val_loader, model, loss_fn, optimizer, train=False)

            print(f'\nEpoch {e + 1}:')
            print_metrics(train_results, 'TRAIN')
            print_metrics(val_results, 'VAL')

            if args.use_gaze:
                model_metrics_folder = os.path.join(fold_metrics_folder, 'gaze')
                if not os.path.exists(model_metrics_folder):
                    os.makedirs(model_metrics_folder)
                    print("Created folder for model metrics")
            else:
                model_metrics_folder = os.path.join(fold_metrics_folder, 'dummy')
                if not os.path.exists(model_metrics_folder):
                    os.makedirs(model_metrics_folder)

            file_name = 'gaze_model_metrics.csv' if args.use_gaze else 'baseline_model_metrics.csv'
            csv_file_path = os.path.join(fold_metrics_folder, file_name)

            # Check if the file exists. If not, write the header.
            if not os.path.exists(csv_file_path):
                with open(csv_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    headers = ['Epoch'] + [f'Train_{k}' for k in train_results.keys()] + [f'Val_{k}' for k in val_results.keys()]
                    writer.writerow(headers)

            # Append the epoch data to the CSV file
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                row = [e + 1] + list(train_results.values()) + list(val_results.values())
                writer.writerow(row)

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

        for metric in train_results:
            train_metrics[metric].append(train_results[metric])
            val_metrics[metric].append(val_results[metric])

        _, _, test_results = iterate(test_loader, model, loss_fn, optimizer, train=False)
        print('\nTest Metrics:')
        print_metrics(test_results, 'TEST')
        test_metrics_row = [fold_idx] + list(test_results.values())
        all_test_metrics.append(test_metrics_row)

        # Save test metrics to CSV
        test_metrics_filename = 'test_metrics_{}.csv'.format('gaze' if args.use_gaze else 'dummy')
        test_metrics_file_path = os.path.join(base_metrics_folder, test_metrics_filename)
        headers = ['Fold'] + [f'Test_{metric}' for metric in test_metrics.keys()]
        write_to_csv(test_metrics_file_path, all_test_metrics, headers, mode='w')

    if best_model_state is not None:
        if args.use_gaze:
            torch.save(best_model_state, 'best_model_gaze.pth')
        else:
            torch.save(best_model_state, 'best_model_dummy.pth')
        model.load_state_dict(best_model_state)

    print('\n\n> 10-fold CV done')
    print_mean_metrics(train_metrics, 'TRAIN')
    print_mean_metrics(val_metrics, 'VAL')
    print_mean_metrics(test_metrics, 'TEST')

    # Writing overall test metrics
    overall_test_metrics_folder = os.path.join(base_metrics_folder, 'test_metrics')
    create_folder_if_not_exists(overall_test_metrics_folder)
    overall_test_metrics_file = os.path.join(overall_test_metrics_folder, 'test_metrics.csv')

    headers = [f'Test_{k}' for k in test_results.keys()] if fold_number == 1 else None
    write_to_csv(overall_test_metrics_file, list(test_results.values()), headers, mode='a')

if __name__ == "__main__":
    main()
