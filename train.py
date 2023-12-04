import sys
import numpy as np
from argparse import ArgumentParser
import torch
from torch import softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR  # Import StepLR from torch.optim.lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model import BertSentimentClassifier
from data import SentimentDataSet

def init_metrics():
    return {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

def print_metrics(metrics, split):
    print(f'\n[{split}]: ', end='')
    for k, v in metrics.items():
        avg_value = np.mean(v) if isinstance(v, list) else v
        print(f' {k}: {avg_value:.2f}', end='')

def get_metrics(targets, predictions, average_method):
    return {
        'accuracy': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average=average_method, zero_division=0),
        'precision': precision_score(targets, predictions, average=average_method, zero_division=0),
        'recall': recall_score(targets, predictions, average=average_method, zero_division=0)
    }

def iterate(dataloader, model, loss_fn, optimizer, l1_lambda=0.001, train=True):
    epoch_loss = 0.0
    all_targets = []
    all_predictions = []

    for batch in dataloader:
        # Extract and check data
        input_ids, attention_mask, et_features, targets = batch['input_ids'], batch['attention_mask'], batch['et_features'], batch['labels']
        for tensor in [input_ids, attention_mask, et_features, targets]:
            if tensor is not None and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                raise ValueError("NaN or Inf in tensor")

        if torch.cuda.is_available():
            input_ids, attention_mask, targets = input_ids.cuda(), attention_mask.cuda(), targets.cuda()
            et_features = et_features.cuda() if et_features is not None else None

        with torch.autograd.set_detect_anomaly(True):
            logits = model(input_ids, attention_mask, et_features)
            loss = loss_fn(logits, targets)

            if l1_lambda > 0:  # Apply L1 regularization
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_penalty

            if train:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()

        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy())
        epoch_loss += loss.item()

    return epoch_loss, get_metrics(all_targets, all_predictions, 'macro' if model.num_labels > 2 else 'binary')


def main():
    parser = ArgumentParser()
    parser.add_argument('--num-sentiments', type=int, default=3, help='2: binary classification, 3: ternary.')
    parser.add_argument('--use-gaze', action='store_true', help='Use gaze features if set')
    parser.add_argument('--word-features-file', type=str, required=True, help='Path to the word level features file')
    args = parser.parse_args()

    dataset = SentimentDataSet('sentiment_labels_task1.csv', args.word_features_file)
    lstm_units = 300 if args.num_sentiments == 2 else 150

    XE_loss = CrossEntropyLoss()
    train_metrics = init_metrics()
    test_metrics = init_metrics()

    best_val_loss = float('inf')
    best_model_state = None

    for k, (train_loader, test_loader) in enumerate(dataset.split_cross_val(10)):
        model = BertSentimentClassifier(lstm_units, args.num_sentiments, args.use_gaze)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add weight_decay for L2 regularization
        for e in range(10):
            train_loss, train_results = iterate(train_loader, model, XE_loss, optimizer)
            test_loss, test_results = iterate(test_loader, model, XE_loss, optimizer, train=False)

            print(f'Epoch {e + 1}:')
            print_metrics(train_results, 'TRAIN')
            print_metrics(test_results, 'TEST')

            if test_loss < best_val_loss:
                best_val_loss = test_loss
                best_model_state = model.state_dict()

            scheduler.step()

    if best_model_state is not None:
        torch.save(best_model_state, 'best_model.pth')

    print('\n\n> 10-fold CV done')
    print_metrics(train_metrics, 'MEAN TRAIN')
    print_metrics(test_metrics, 'MEAN TEST')

if __name__ == "__main__":
    main()