import os
from pathlib import Path
import numpy as np
import pickle

from dataset.dataset import Dataset, HyperParameterSet
from dataset.impl.lowlevel import Sample, LowLevelDataset
from utils.logger import log


# ----------------------------------------------------------------------------------------------------------------------
def silence_masking(arr, val):
    """
    Replace silence (nan) with <val> in <arr> and
    create new feature (column) indicating whether it was
    silent or not
    """
    arr = np.asarray(arr)
    nan_mask = np.isnan(arr)
    replaced = np.where(nan_mask, val, arr)
    indicator = nan_mask.astype(int)
    return np.column_stack((replaced, indicator))


class DatasetSaraga(Dataset):
    def __init__(self, root="data/melakarta/pretrain", num_synth=0, augment=False, svara_form_target=False, svara_form_group=False):
        self.num_synth = num_synth
        self.augment = augment
        self.svara_form_target = svara_form_target
        self.svara_form_group = svara_form_group
        root = "data/melakarta/pretrain/augment" if augment else "data/melakarta/pretrain/no_augment"
        super(DatasetSaraga, self).__init__("Bhairavi", root, num_synth)

    def _load_underlying_dataset(self):
        self.num_features = 2
        self.num_folds = 1
        self.underlying_dataset = self._load_bhairavi()
        
    def get_hyperparameter_set(self):
        return HyperParameterSet(learning_rate=0.001,
                                 batch_size=2048,
                                 weight_decay=0.001,
                                 num_epochs=150)

    def _get_augmenters(self, random_seed):
        return []#[AugTransposition(1, 2, random_seed), AugTimestretch(1, 5, random_seed, 0.95, 1.05)]

    def _load_bhairavi(self):
        """
        Loads the Bhairavi dataset.
        """
        # svara, svara_form, prec, curr, succ
        train_path = os.path.join(self.root, 'TRAIN.pkl')
        test_path = os.path.join(self.root, 'TEST.pkl')
        labels_path = os.path.join(self.root, 'LABELS.pkl')
        
        train = load_pkl(train_path)
        test = load_pkl(test_path)
        labels = load_pkl(labels_path)
        self.labels = labels

        # extract time series and convert to two dimensional masking silence
        svara_labels_train = [int(x[0]) for x in train]
        svara_form_labels_train = [int(x[1]) for x in train]
        prec_train_data = [silence_masking(x[2], -10000) for x in train]
        curr_train_data = [silence_masking(x[3], -10000) for x in train]
        succ_train_data = [silence_masking(x[4], -10000) for x in train]
        
        svara_labels_test = [int(x[0]) for x in test]
        svara_form_labels_test = [int(x[1]) for x in test]
        prec_test_data = [silence_masking(x[2], -10000) for x in test]
        curr_test_data = [silence_masking(x[3], -10000) for x in test]
        succ_test_data = [silence_masking(x[4], -10000) for x in test]

        # (e.g. train[0] test[0] mean test on FOLD[0], train on everything else)
        self.train_indices = [[] for i in range(self.num_folds)] # list of folds(list of indices)
        self.test_indices = [[] for i in range(self.num_folds)]  # list of folds(list of indices)

        samples = []

        train_target = svara_form_labels_train if self.svara_form_target else svara_labels_train
        test_target = svara_form_labels_test if self.svara_form_target else svara_labels_test

        train_group = svara_form_labels_train if self.svara_form_group else svara_labels_train
        test_group = svara_form_labels_test if self.svara_form_group else svara_labels_test

        # create train samples
        train_generator = enumerate(zip(
            train_target, train_group, 
            prec_train_data, curr_train_data, succ_train_data))
        
        for i, (slabel, sflabel, prec, curr, succ) in train_generator:

            samples += [Sample(slabel, sflabel, prec, curr, succ, '', train_path)]
            self.train_indices[i%self.num_folds].append(i)
        
        # create test samples
        test_generator = enumerate(zip(
            test_target, test_group, 
            prec_test_data, curr_test_data, succ_test_data), i+1)

        for j, (slabel, sflabel, prec, curr, succ) in test_generator:

            samples += [Sample(slabel, sflabel, prec, curr, succ, '', test_path)]
            self.test_indices[i%self.num_folds].append(j)
        
        return LowLevelDataset(samples, self.train_indices, self.test_indices)


def load_pkl(path):
    file = open(path,'rb')
    return pickle.load(file)

# if len(data[0].shape)==1: # one dimensional vector
#   data = np.array([np.reshape(d, (-1, 1)) for d in data])
    