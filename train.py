import sys
import random
import numpy as np
from argparse import ArgumentParser
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.nn import BCELoss, CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from model import BertSentimentClassifier
from data import SentimentDataSet

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    print(f'{prefix} Accuracy: {metrics["accuracy"]:.4f}, Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f},F1: {metrics["f1"]:.4f}, ROC-AUC: {metrics["roc_auc"]:.4f}')
    return 

def iterate(dataloader, model, loss_fn, optimizer, l1_lambda=0.001, train=True):
    epoch_loss = 0.0
    all_targets = []
    all_predictions = []
    all_scores = []

    for batch in dataloader:
        # Extract and check data
        input_ids, attention_mask, et_features, targets = batch['input_ids'], batch['attention_mask'], batch['et_features'], batch['labels']
        if torch.cuda.is_available():
            input_ids, attention_mask, targets = input_ids.cuda(), attention_mask.cuda(), targets.cuda()
            et_features = et_features.cuda() if et_features is not None else None

        with torch.autograd.set_detect_anomaly(True):
            logits = model(input_ids, attention_mask, et_features)

         
            # Squeeze logits to match the shape of targets
            logits = logits.squeeze()
            # Targets should also be a float tensor with the same shape
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
    lstm_units = 300 if args.num_sentiments == 2 else 150

    loss_fn = BCELoss() if args.num_sentiments == 2 else CrossEntropyLoss()
    train_metrics = init_metrics()
    test_metrics = init_metrics()

    best_val_loss = float('inf')
    best_model_state = None

    for k, (train_loader, test_loader) in enumerate(dataset.split_cross_val(5)):
        model = BertSentimentClassifier(lstm_units, args.num_sentiments, args.use_gaze)
        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.90)

        for e in range(10):
            train_loss, train_scores, train_results = iterate(train_loader, model, loss_fn, optimizer)
            test_loss, test_scores, test_results = iterate(test_loader, model, loss_fn, optimizer, train=False)

            print(f'\nEpoch {e + 1}:')
            print_metrics(train_results, 'TRAIN')
            print_metrics(test_results, 'TEST')

            for metric in train_results:
                train_metrics[metric].append(train_results[metric])
                test_metrics[metric].append(test_results[metric])

            if test_loss < best_val_loss:
                best_val_loss = test_loss
                best_model_state = model.state_dict()

            scheduler.step()

    if best_model_state is not None:
        torch.save(best_model_state, 'best_model.pth')

    print('\n\n> 10-fold CV done')
    print_metrics({'accuracy': np.mean(train_metrics['accuracy']), 
                   'precision': np.mean(train_metrics['precision']), 
                   'recall': np.mean(train_metrics['recall']), 
                   'f1': np.mean(train_metrics['f1']),
                   'roc_auc': np.mean(train_metrics['roc_auc'])}, 'MEAN TRAIN')
    print_metrics({'accuracy': np.mean(test_metrics['accuracy']), 
                   'precision': np.mean(test_metrics['precision']), 
                   'recall': np.mean(test_metrics['recall']), 
                   'f1': np.mean(test_metrics['f1']),
                   'roc_auc': np.mean(test_metrics['roc_auc'])}, 'MEAN TEST')

if __name__ == "__main__":
    main()
