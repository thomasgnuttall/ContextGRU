import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
import time
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
import hashlib
import torch
import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import csv

from model import DeepGRU
from dataset.datafactory import DataFactory
from utils.average_meter import AverageMeter  
from utils.logger import log
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='DeepGRU Training with Triplet Loss')
parser.add_argument('--dataset', metavar='DATASET_NAME',
                    choices=DataFactory.dataset_names,
                    default='saraga')
parser.add_argument('--seed', type=int, default=1570254494)
parser.add_argument('--use-cuda', action='store_true', default=True)
parser.add_argument('--run-name', type=str, default=None)
parser.add_argument('--save-model', action='store_true', default=False)
parser.add_argument('--augment', action='store_true', default=False)
parser.add_argument('--melodic-context', action='store_true', default=False)
parser.add_argument('--svara-form-target', action='store_true', default=False)
parser.add_argument('--svara-form-group', action='store_true', default=False)
args = parser.parse_args()

seed = int(time.time()) if args.seed == -1 else args.seed
use_cuda = torch.cuda.is_available() and args.use_cuda


def remove_file_if_exists(path):
    """
    Removes the file at the given path if it exists.
    
    Args:
        path (str): The path to the file to be removed.
    """
    if os.path.isfile(path):
        os.remove(path)


def append_tuple_to_csv(path, row):
    """
    Appends a tuple as a row to a CSV file at the given path.
    
    Args:
        path (str): The path to the CSV file.
        row (Tuple): The row to append, as a tuple of values.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def create_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)

def cpath(*args):
    """
    Wrapper around os.path.join, create path concatenating args and
    if the containing directories do not exist, create them.
    """
    path = os.path.join(*args)
    create_if_not_exists(path)
    return path

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


def get_hashable_key(arr):
    """Generate a unique hash for a NumPy array."""

    arr_int = np.round(arr).astype(int)

    arr_bytes = arr_int.tobytes()  # Convert to raw bytes
    return hashlib.sha256(arr_bytes).hexdigest()  # Generate SHA-256 hash


def plot_time_series(series1, series2, series3, path, labels=("Series 1", "Series 2", "Series 3")):
    """
    Plots three time series on separate subplots in the same figure.

    Parameters:
    - series1, series2, series3: array-like, time series data
    - labels: tuple of strings, labels for the three series (default names used)
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

    for ax, series, label in zip(axes, [series1, series2, series3], labels):
        ax.plot(np.arange(len(series)), series, label=label, linewidth=2)
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Index")
    plt.tight_layout()
    plt.savefig(path)
    plt.close('all')


def sample_triplets(embeddings, groups):
    batch_size = embeddings.shape[0]
    anchor = embeddings
    positive = torch.zeros_like(anchor)
    negative = torch.zeros_like(anchor)
    mask = [True]*len(anchor)
    
    for i in range(len(embeddings)):
        group = groups[i]
        
        pos_ix = [j for j,g in enumerate(groups) if g == group]
        neg_ix = [j for j,g in enumerate(groups) if g != group]

        if not pos_ix or not neg_ix:
            positive[i] = np.array([])
            negative[i] = np.array([])
            mask[i] = False

        positive[i] = embeddings[np.random.choice(pos_ix)]
        negative[i] = embeddings[np.random.choice(neg_ix)]

    anchor = anchor[mask]
    positive = positive[mask]
    negative = negative[mask]

    return anchor, positive, negative


def run_batch(batch, model, triplet_criterion):
    prec_examples, curr_examples, succ_examples, prec_lengths, curr_lengths, succ_lengths, _, groups = batch
    
    if use_cuda:
        prec_examples = prec_examples.cuda()
        curr_examples = curr_examples.cuda()
        succ_examples = succ_examples.cuda()

    normalized_embeddings = model(
        prec_examples, curr_examples, succ_examples, 
        prec_lengths, curr_lengths, succ_lengths, contrastive=True)

    anchor, positive, negative = sample_triplets(normalized_embeddings, groups)

    loss = triplet_criterion(anchor, positive, negative)

    return 100.0, curr_examples.size(0), loss  # Placeholder accuracy


def run_fold(dataset, fold_idx, results_path, use_cuda):
    hyperparameters = dataset.get_hyperparameter_set()
    n_classes = 7 if not args.svara_form_target else 78
    model = DeepGRU(dataset.num_features, n_classes, with_context=args.melodic_context)
    log(f"  number of classes: {n_classes}")
    triplet_criterion = TripletLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.learning_rate, weight_decay=hyperparameters.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    train_loader, test_loader = dataset.get_data_loaders(fold_idx, shuffle=True, random_seed=seed + fold_idx, normalize=False)

    dtw_cache = {}
    encode_cache = {}
    best_test_loss = float('inf')  # Track the lowest test loss
    patience = 10
    no_improve_epochs = 0

    row = ("fold_idx", "epoch", "train_loss", "test_loss")
    append_tuple_to_csv(results_path, row)
    for epoch in range(hyperparameters.num_epochs):
        log(str(datetime.datetime.now()))
        loss_meter = AverageMeter()
        train_meter = AverageMeter()

        # Training Loop
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            accuracy, curr_batch_size, loss = run_batch(batch, model, triplet_criterion)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), curr_batch_size)
            train_meter.update(accuracy, curr_batch_size)
        
        train_accuracy = train_meter.avg
        avg_train_loss = loss_meter.avg

        log(f'Epoch: [{epoch}]  [Avg Train Loss] {loss_meter.avg:.6f}')

        # Evaluation Loop (Test)
        model.eval()
        with torch.no_grad():
            test_meter = AverageMeter()
            test_loss_meter = AverageMeter()

            for batch in test_loader:
                accuracy, curr_batch_size, test_loss = run_batch(batch, model, triplet_criterion)
                test_meter.update(accuracy, curr_batch_size)
                test_loss_meter.update(test_loss.item(), curr_batch_size)

            test_accuracy = test_meter.avg
            avg_test_loss = test_loss_meter.avg  # Average test loss for the epoch
            
            log(f'                  [Avg Test Loss] {test_loss_meter.avg:.6f}')

        # Early Stopping Logic Based on Test Loss
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            no_improve_epochs = 0  # Reset patience counter
            if args.save_model:
                model_path = cpath(f"models/{args.run_name}/best_model_fold={fold_idx}" if args.run_name else f"models/best_model_fold={fold_idx}")
                log(f"saving model to {model_path}")
                torch.save(model.state_dict(), model_path)
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            log(f"Early stopping triggered after {epoch} epochs due to no improvement in test loss.")
            break
        
        scheduler.step(test_loss)

        row = (fold_idx, epoch, avg_train_loss, avg_test_loss)
        append_tuple_to_csv(results_path, row)

    return best_test_loss


def main():
    
    log.set_dataset_name(args.dataset)
    
    dataset = DataFactory.instantiate(
        args.dataset, augment=args.augment,
        svara_form_target=args.svara_form_target, 
        svara_form_group=args.svara_form_group, 
        num_synth=0)
    
    if args.augment:
        log("Using augmented data")
    
    if args.svara_form_target:
        log("Target: svara form")
    else:
        log("Target: svara")

    if args.svara_form_group:
        log("Triplets informed by propagated svara form")
    else:
        log("Triplets informed by propagated svara")

    log.log_dataset(dataset)
    log("Random seed: " + str(seed))
    torch.manual_seed(seed)
    accuracies = []
    results_path = cpath(f"models/{args.run_name}/results.csv")
    remove_file_if_exists(results_path)
    for fold_idx in range(dataset.num_folds):
        log(f'Running fold "{fold_idx}"...')
        test_accuracy = run_fold(dataset, fold_idx, results_path, use_cuda)
        accuracies.append(test_accuracy)
        log(f'Fold "{fold_idx}" complete, final accuracy: {test_accuracy}')
    log('-----------------------------------------------------------------------')
    log(f'Training complete! Average accuracy: {np.mean(accuracies)}')

if __name__ == '__main__':
    main()
