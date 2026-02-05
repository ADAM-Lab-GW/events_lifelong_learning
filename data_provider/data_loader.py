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
    elif name.upper() in ["EVENTSYM", "EVENTSYM3"]:
        config = DATASET_CONFIGS.get('eventsym', None)

        train_root = os.path.join(data_dir, "eventSym", "training")
        test_root  = os.path.join(data_dir, "eventSym", "testing")

        # Build hierarchical datasets
        if not only_test:
            eventsym_train = HierarchicalNpyDataset(train_root, verbose=verbose)
        eventsym_test = HierarchicalNpyDataset(test_root, verbose=verbose)

        n_subclasses = len(eventsym_test.sub_to_id)
        n_main = len(eventsym_test.main_to_id)

        if tasks > n_subclasses:
            raise ValueError(f"Experiment 'eventSym' cannot have more than {n_subclasses} tasks!")

        classes_per_task = int(np.floor(n_subclasses / tasks)) if tasks > 0 else n_subclasses

        if not only_config:
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id)
                for task_id in range(tasks)
            ]

            # split into sub-tasks (SubDataset must filter using y_sub)
            train_datasets, test_datasets = [], []
            for labels in labels_per_task:
                if not only_test:
                    train_datasets.append(SubDataset(eventsym_train, labels, target_transform=None))
                test_datasets.append(SubDataset(eventsym_test, labels, target_transform=None))




    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)


# def load_ncaltech(train, dir_name, target_transform=None):
#     if train:
#         train_data_path = os.path.join(dir_name, 'training')
#         dataset = torchvision.datasets.DatasetFolder(root=train_data_path,
#                                                      loader=ReadAsynetFile(),
#                                                      extensions=(".npy",),
#                                                      target_transform=target_transform)
#     else:
#         test_data_path = os.path.join(dir_name, 'testing')
#         dataset = torchvision.datasets.DatasetFolder(root=test_data_path,
#                                                      loader=ReadAsynetFile(),
#                                                      extensions=(".npy",),
#                                                      target_transform=target_transform)

#     return dataset

def load_ncaltech(train, dir_name, target_transform=None):
    if train:
        train_data_path = os.path.join(dir_name, 'training')
        # EventSym (hierarchical) lives in: .../eventSym/training/main/sub/*.npy
        if os.path.basename(dir_name).lower() == "eventsym":
            dataset = HierarchicalNpyDataset(train_data_path)
        else:
            dataset = torchvision.datasets.DatasetFolder(
                root=train_data_path,
                loader=ReadAsynetFile(),
                extensions=(".npy",),
                target_transform=target_transform
            )
    else:
        test_data_path = os.path.join(dir_name, 'testing')
        if os.path.basename(dir_name).lower() == "eventsym":
            dataset = HierarchicalNpyDataset(test_data_path)
        else:
            dataset = torchvision.datasets.DatasetFolder(
                root=test_data_path,
                loader=ReadAsynetFile(),
                extensions=(".npy",),
                target_transform=target_transform
            )
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
    
import glob

class HierarchicalNpyDataset(torch.utils.data.Dataset):
    """
    Expects:
      root/
        main_1/
          sub_1/*.npy
          sub_2/*.npy
        main_2/
          sub_3/*.npy
          ...

    Returns: x, y_sub, y_main
    Also exposes mappings:
      main_to_id, sub_to_id, sub_to_main (dict[int->int])
    """
    def __init__(self, root_dir, verbose=False):
        self.root_dir = root_dir
        self.samples = []  # list of (filepath, y_sub, y_main)

        # Deterministic ordering
        main_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.main_to_id = {m: i for i, m in enumerate(main_names)}

        # Build subclass ids globally across all mains
        sub_names_global = []
        for m in main_names:
            m_dir = os.path.join(root_dir, m)
            sub_names = sorted([d for d in os.listdir(m_dir) if os.path.isdir(os.path.join(m_dir, d))])
            for s in sub_names:
                sub_names_global.append((m, s))

        self.sub_to_id = { (m, s): i for i, (m, s) in enumerate(sub_names_global) }
        self.sub_to_main = { self.sub_to_id[(m, s)]: self.main_to_id[m] for (m, s) in sub_names_global }

        # Collect files
        for (m, s) in sub_names_global:
            y_main = self.main_to_id[m]
            y_sub = self.sub_to_id[(m, s)]
            sub_dir = os.path.join(root_dir, m, s)
            for fp in sorted(glob.glob(os.path.join(sub_dir, "*.npy"))):
                self.samples.append((fp, y_sub, y_main))

        if verbose:
            print(f" --> HierarchicalNpyDataset: {len(self.samples)} samples, "
                  f"{len(self.main_to_id)} main classes, {len(self.sub_to_id)} subclasses")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, y_sub, y_main = self.samples[idx]
        x = np.load(fp)
        x = torch.from_numpy(x)

        x = torch.from_numpy(np.load(fp)).float()
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"NaN/Inf in input: {fp}")

        return x, int(y_sub), int(y_main)

