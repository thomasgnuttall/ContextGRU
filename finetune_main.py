import argparse
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import csv

from model import DeepGRU
from dataset.datafactory import DataFactory
from utils.average_meter import AverageMeter  # Running average computation
from utils.logger import log                  # Logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeepGRU Training')
parser.add_argument('--dataset', metavar='DATASET_NAME',
                    choices=DataFactory.dataset_names,
                    help='dataset to train on: ' + ' | '.join(DataFactory.dataset_names),
                    default='bhairavi')
parser.add_argument('--seed', type=int, metavar='N',
                    help='random number generator seed, use "-1" for random seed',
                    default=1570254494)
parser.add_argument('--num-synth', type=int, metavar='N',
                    help='number of synthetic samples to generate',
                    default=0)
parser.add_argument('--use-cuda', action='store_true',
                    help='use CUDA if available',
                    default=True)
parser.add_argument('--run-name', type=str,
                    help='unique name for run',
                    default=None)
parser.add_argument('--pretrain-path', type=str,
                    help='unique name for run',
                    default='models/saraga/best_model_fold=0')
parser.add_argument('--save-model', action='store_true',
                    help='Save best model',
                    default=False)
parser.add_argument('--augment', action='store_true', default=False)
parser.add_argument('--melodic-context', action='store_true', default=False)
parser.add_argument('--svara-form-target', action='store_true', default=False)
parser.add_argument('--svara-form-group', action='store_true', default=False)

# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
def main():
    # Load the dataset
    log.set_dataset_name(args.dataset)
    dataset = DataFactory.instantiate(
        args.dataset, augment=args.augment, 
        svara_form_target=args.svara_form_target, 
        svara_form_group=args.svara_form_group, 
        num_synth=0)
    log.log_dataset(dataset)
    log("Random seed: " + str(seed))
    torch.manual_seed(seed)
    
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

    # Run each fold and average the results
    accuracies = []
    results_path = cpath(f"models/{args.run_name}/results.csv")
    remove_file_if_exists(results_path)
    for fold_idx in range(dataset.num_folds):
        log('Running fold "{}"...'.format(fold_idx))
        test_accuracy = run_fold(dataset, fold_idx, results_path, use_cuda)
        accuracies += [test_accuracy]

        log('Fold "{}" complete, final accuracy: {}'.format(fold_idx, test_accuracy))

    log('')
    log('-----------------------------------------------------------------------')
    log('Training complete!')
    log('Average F1: {}'.format(np.mean(accuracies)))


# ----------------------------------------------------------------------------------------------------------------------
def run_fold(dataset, fold_idx, results_path, use_cuda):
    """
    Trains/tests the model on the given fold
    """

    hyperparameters = dataset.get_hyperparameter_set()

    log('Load pretrained model')
    n_classes = 7 if not args.svara_form_target else 78

    # Load pretrained model
    pretrained_model = DeepGRU(dataset.num_features, n_classes, with_context=args.melodic_context)
    
    pretrained_model_path = args.pretrain_path
    
    log('Extract state dict')
    state_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}

    pretrained_model.load_state_dict(state_dict)
    
    log('Initialize current model')
    # Initialize fine-tuning model
    model = DeepGRU(dataset.num_features, n_classes, with_context=args.melodic_context)
    log(f"  number of classes: {n_classes}")

    # Load pretrained encoder weights (ignore classifier)
    pretrained_dict = pretrained_model.state_dict()
    finetune_dict = model.state_dict()

    # Filter out classifier weights
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "classifier" not in k}
    
    log('Update state dict')
    # Update fine-tune model with pretrained encoder weights
    finetune_dict.update(pretrained_dict)
    model.load_state_dict(finetune_dict)

    # FINETUNING!
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    # Create data loaders
    train_loader, test_loader = dataset.get_data_loaders(fold_idx,
                                                         shuffle=True,
                                                         random_seed=seed+fold_idx,
                                                         normalize=False)

    best_train_accuracy = -float('inf')
    best_test_accuracy = -float('inf')
    
    row = ("fold_idx", "epoch", "avg_train_loss", "avg_test_loss", "avg_train_f1","avg_test_f1")
    append_tuple_to_csv(results_path, row)

    # Phase 1: Freeze encoder, train classifier
    classifier_epochs = 50
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=hyperparameters.learning_rate,
                                 weight_decay=hyperparameters.weight_decay)


    log('Phase 1 - Train classifier only')
    log('-------------------------------')
    # Train the model
    for preepoch in range(classifier_epochs):
        loss_meter = AverageMeter()
        train_meter = AverageMeter()

        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            accuracy, curr_batch_size, loss = run_batch(batch, model, criterion)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), curr_batch_size)
            train_meter.update(accuracy, curr_batch_size)

        train_accuracy = train_meter.avg
        train_loss = loss_meter.avg

        row = (fold_idx, preepoch, train_loss, None, train_accuracy, None)
        append_tuple_to_csv(results_path, row)
        
        log(f'Epoch: [{preepoch}]')
        log(f'       [Train Loss]          {train_loss}')
        log(f'       [Train F1]      {train_accuracy}')


    log('Phase 2 - Train entire network')
    log('-------------------------------')
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hyperparameters.learning_rate * 0.1,  # Typically a lower LR
                                 weight_decay=hyperparameters.weight_decay)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Train the model
    for epoch in range(preepoch + 1, hyperparameters.num_epochs):
        loss_meter = AverageMeter()
        train_meter = AverageMeter()
        test_meter = AverageMeter()

        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            accuracy, curr_batch_size, loss = run_batch(batch, model, criterion)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Update stats
            loss_meter.update(loss.item(), curr_batch_size)
            train_meter.update(accuracy, curr_batch_size)

        train_accuracy = train_meter.avg
        train_loss = loss_meter.avg

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy

        model.eval()
        with torch.no_grad():
            test_loss_meter = AverageMeter()

            for batch in test_loader:

                accuracy, curr_batch_size, loss = run_batch(batch, model, criterion)
                test_loss_meter.update(loss.item(), curr_batch_size)
                test_meter.update(accuracy, curr_batch_size)

            test_accuracy = test_meter.avg
            test_loss = test_loss_meter.avg

            # Update best accuracies
            if best_test_accuracy < test_accuracy:
                best_test_accuracy = test_accuracy
                if args.save_model:
                    if args.run_name:
                        run_name = args.run_name
                        model_path = cpath("models", run_name, f"best_model_fold={fold_idx}")
                    else:
                        model_path = cpath("models", f"best_model_fold={fold_idx}")
                    log(f"saving model to {model_path}")
                    torch.save(model.state_dict(), model_path)
        
        scheduler.step(test_loss)

        row = (fold_idx, epoch, train_loss, test_loss, train_accuracy, test_accuracy)
        append_tuple_to_csv(results_path, row)
        
        log(f'Epoch: [{epoch}]')
        log(f'       [Train Loss]          {train_loss}')
        log(f'       [Train F1]      {train_accuracy}')
        log(f'       [Test Loss]           {test_loss}')
        log(f'       [Test F1]       {test_accuracy}')
 
        if best_test_accuracy == 100:
            break

    return best_test_accuracy



def f1_score(logits, labels):

    predicted = logits.argmax(1)
    num_classes = logits.size(1)
    f1_scores = []

    for c in range(num_classes):
        true_positives = ((predicted == c) & (labels == c)).sum().item()
        predicted_positives = (predicted == c).sum().item()
        actual_positives = (labels == c).sum().item()

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / num_classes


# ----------------------------------------------------------------------------------------------------------------------
def run_batch(batch, model, criterion):
    """
    Runs the forward pass on a batch and computes the loss and accuracy
    """
    prec_examples, curr_examples, succ_examples, prec_lengths, curr_lengths, succ_lengths, labels, groups = batch

    if use_cuda:
        prec_examples = prec_examples.cuda()
        curr_examples = curr_examples.cuda()
        succ_examples = succ_examples.cuda()
        labels = labels.cuda()
        
    # Forward and loss computation
    logits = model(
        prec_examples, curr_examples, succ_examples, 
        prec_lengths, curr_lengths, succ_lengths)

    loss = criterion(logits, labels)

    # Compute the accuracy
    #predicted = logits.argmax(1)
    #correct = (predicted == labels).sum().item()
    curr_batch_size = labels.size(0)
    #accuracy = correct / curr_batch_size * 100.0
    
    f1 = f1_score(logits, labels)

    return f1, curr_batch_size, loss


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
