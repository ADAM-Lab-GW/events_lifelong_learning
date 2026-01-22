"""
The structure of this file and some code are similar to
https://github.com/GMvandeVen/brain-inspired-replay/blob/master/data/load.py

Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

Some settings for datasets.
"""

import copy
import os

import numpy as np
import torch
import torchvision
from torch.utils.data import ConcatDataset
from torchvision import transforms

from data_provider import utils
from data_provider.available import DATASET_CONFIGS
from data_provider.manipulate import SubDataset


def get_dataset(name, type_='train', capacity=None, directory='./store/datasets',
                verbose=False, target_transform=None):
    if name.lower() == utils.DatasetNames.NCALTECH12.value:
        dataset = load_ncaltech(train=False if type_ == 'test' else True,
                                dir_name='{dir}/ncaltech12'.format(dir=directory),
                                target_transform=target_transform)
    elif name.lower() == utils.DatasetNames.NCALTECH256.value:
        dataset = load_ncaltech(train=False if type_ == 'test' else True,
                                dir_name='{dir}/ncaltech256'.format(dir=directory),
                                target_transform=target_transform)
    elif name.lower() == utils.DatasetNames.NCALTECH101.value:
        dataset = load_ncaltech(train=False if type_ == 'test' else True,
                                dir_name='{dir}/ncaltech101'.format(dir=directory),
                                target_transform=target_transform)
    elif name.lower() == utils.DatasetNames.NMNIST.value:
        dataset = load_ncaltech(train=False if type_ == 'test' else True,
                                dir_name='{dir}/nmnist'.format(dir=directory),
                                target_transform=target_transform)
    elif name.lower() == utils.DatasetNames.EVENTSYM.value:
        dataset = load_ncaltech(train=False if type_ == 'test' else True,
                                dir_name='{dir}/eventSym'.format(dir=directory),
                                target_transform=target_transform)
    else:
        raise ValueError("Check code")

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type_, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


##-------------------------------------------------------------------------------------------------------------------##


def get_data_batch_strategy(name, data_dir="./store/datasets", verbose=False):
    # Define data-type
    if name == "NCALTECH12":
        data_type = 'ncaltech12'
    elif name == "NCALTECH256":
        data_type = 'ncaltech256'
    elif name == "NCALTECH101":
        data_type = 'ncaltech101'
    elif name == "NMNIST":
        data_type = 'nmnist'
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    # Get config-dict and data-sets
    config = DATASET_CONFIGS[data_type]
    trainset = get_dataset(data_type, type_='train', directory=data_dir, verbose=verbose)
    testset = get_dataset(data_type, type_='test', directory=data_dir, verbose=verbose)

    # Return tuple of data-sets and config-dictionary
    return (trainset, testset), config


def get_data_incremental_strategy(name, tasks, data_dir="./store/datasets",
                                  only_config=False, verbose=False, only_test=False):
    if name == 'NCALTECH12':
        # check for number of tasks
        if tasks > 12:
            raise ValueError("Experiment 'NCALTECH12' cannot have more than 12 tasks!")
        # configurations
        config = DATASET_CONFIGS['ncaltech12']
        classes_per_task = int(np.floor(12 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.random.permutation(list(range(12)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                ncaltech12_train = get_dataset('ncaltech12', type_="train", directory=data_dir,
                                               verbose=verbose, target_transform=target_transform)
            ncaltech12_test = get_dataset('ncaltech12', type_="test", directory=data_dir,
                                          verbose=verbose, target_transform=target_transform)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = None
                if not only_test:
                    train_datasets.append(SubDataset(ncaltech12_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(ncaltech12_test, labels, target_transform=target_transform))
    elif name == 'NCALTECH256':
        # check for number of tasks
        if tasks > 257:
            raise ValueError("Experiment 'NCALTECH256' cannot have more than 257 tasks!")
        # configurations
        config = DATASET_CONFIGS['ncaltech256']
        classes_per_task = int(np.floor(257 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.random.permutation(list(range(257)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                ncaltech256_train = get_dataset('ncaltech256', type_="train", directory=data_dir,
                                                verbose=verbose, target_transform=target_transform)
            ncaltech256_test = get_dataset('ncaltech256', type_="test", directory=data_dir,
                                           verbose=verbose, target_transform=target_transform)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = None
                if not only_test:
                    train_datasets.append(SubDataset(ncaltech256_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(ncaltech256_test, labels, target_transform=target_transform))
    elif name == 'NCALTECH101':
        # check for number of tasks
        if tasks > 101:
            raise ValueError("Experiment 'NCALTECH101' cannot have more than 101 tasks!")
        # configurations
        config = DATASET_CONFIGS['ncaltech101']
        classes_per_task = int(np.floor(101 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.random.permutation(list(range(101)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                ncaltech101_train = get_dataset('ncaltech101', type_="train", directory=data_dir,
                                                verbose=verbose, target_transform=target_transform)
            ncaltech101_test = get_dataset('ncaltech101', type_="test", directory=data_dir,
                                           verbose=verbose, target_transform=target_transform)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = None
                if not only_test:
                    train_datasets.append(SubDataset(ncaltech101_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(ncaltech101_test, labels, target_transform=target_transform))
    elif name == 'NMNIST':
        # check for number of tasks
        if tasks > 10:
            raise ValueError("Experiment 'NMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['nmnist']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                nmnist_train = get_dataset('nmnist', type_="train", directory=data_dir,
                                                verbose=verbose, target_transform=target_transform)
            nmnist_test = get_dataset('nmnist', type_="test", directory=data_dir,
                                           verbose=verbose, target_transform=target_transform)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = None
                if not only_test:
                    train_datasets.append(SubDataset(nmnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(nmnist_test, labels, target_transform=target_transform))
    elif name.lower() in ["eventsym", "eventSym".lower()]:
        # configurations
        config = DATASET_CONFIGS["eventSym"]

        if not only_config:
            # ---- load datasets first so we can infer number of labels robustly ----
            # (We load test even if only_test=True, like the other blocks.)
            # target_transform applied after permutation is built.

            # temporary load to get label count if you don't store it in config
            tmp_train = None
            if not only_test:
                tmp_train = get_dataset("eventSym", type_="train", directory=data_dir,
                                        verbose=verbose, target_transform=None)

            tmp_test = get_dataset("eventSym", type_="test", directory=data_dir,
                                verbose=verbose, target_transform=None)

            # Determine number of classes (subclasses) for CL
            # Prefer dataset-provided nr_classes if available, else infer from labels
            if hasattr(tmp_test, "nr_classes"):
                n_classes = int(tmp_test.nr_classes)
            elif tmp_train is not None and hasattr(tmp_train, "nr_classes"):
                n_classes = int(tmp_train.nr_classes)
            else:
                # last-resort inference (slower): scan labels
                labels_all = []
                if tmp_train is not None:
                    labels_all += list(getattr(tmp_train, "labels", []))
                labels_all += list(getattr(tmp_test, "labels", []))
                n_classes = int(len(set(labels_all)))

            # check for number of tasks
            if tasks > n_classes:
                raise ValueError(f"Experiment 'eventSym' cannot have more than {n_classes} tasks!")

            classes_per_task = int(np.floor(n_classes / tasks))
            if classes_per_task < 1:
                raise ValueError(
                    f"Too many tasks ({tasks}) for eventSym with {n_classes} classes "
                    f"(classes_per_task becomes {classes_per_task})."
                )

            # ---- permutation to shuffle label ids (seed-dependent externally) ----
            permutation = np.random.permutation(list(range(n_classes)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(x[y]))

            # ---- load train/test datasets with permutation transform ----
            if not only_test:
                eventSym_train = get_dataset("eventSym", type_="train", directory=data_dir,
                                            verbose=verbose, target_transform=target_transform)
            eventSym_test = get_dataset("eventSym", type_="test", directory=data_dir,
                                        verbose=verbose, target_transform=target_transform)

            # ---- generate labels-per-task ----
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id)
                for task_id in range(tasks)
            ]

            # Handle remainder labels (if n_classes not divisible by tasks):
            remainder = n_classes - classes_per_task * tasks
            if remainder > 0:
                labels_per_task[-1] = labels_per_task[-1] + list(
                    np.array(range(remainder)) + classes_per_task * tasks
                )

            # ---- split into sub-tasks ----
            train_datasets, test_datasets = [], []
            for labels in labels_per_task:
                # After SubDataset filtering, we usually do NOT need an extra target_transform
                # because dataset already permuted labels. Keep None (same as your NCALTECH blocks).
                if not only_test:
                    train_datasets.append(SubDataset(eventSym_train, labels, target_transform=None))
                test_datasets.append(SubDataset(eventSym_test, labels, target_transform=None))

    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task * tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)


def load_ncaltech(train, dir_name, target_transform=None):
    if train:
        train_data_path = os.path.join(dir_name, 'training')
        dataset = torchvision.datasets.DatasetFolder(root=train_data_path,
                                                     loader=ReadAsynetFile(),
                                                     extensions=(".npy",),
                                                     target_transform=target_transform)
    else:
        test_data_path = os.path.join(dir_name, 'testing')
        dataset = torchvision.datasets.DatasetFolder(root=test_data_path,
                                                     loader=ReadAsynetFile(),
                                                     extensions=(".npy",),
                                                     target_transform=target_transform)

    return dataset


class ReadAsynetFile(object):
    """
    Converts an ASYNET file to a tensor.
    """
    def __call__(self, filename):
        """
        Converts
        :param filename:
        :return:
        """
        return self.read(filename)

    @staticmethod
    def read(filename):
        """
        Reads a file
        :param filename: a name of a file.
        :return: a tensor.
        """
        """"""
        data = np.load(filename)
        data = torch.from_numpy(data)

        return data
