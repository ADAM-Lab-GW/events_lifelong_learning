# """
# The code in this file is copied from
# https://github.com/GMvandeVen/brain-inspired-replay/blob/master/data/manipulate.py

# """

# from torch.utils.data import Dataset


# class SubDataset(Dataset):
#     '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

#     After this selection of samples has been made, it is possible to transform the target-labels,
#     which can be useful when doing continual learning with fixed number of output units.'''

#     def __init__(self, original_dataset, sub_labels, target_transform=None):
#         super().__init__()
#         self.dataset = original_dataset
#         self.sub_indeces = []
#         for index in range(len(self.dataset)):
#             if hasattr(original_dataset, "targets"):
#                 if self.dataset.target_transform is None:
#                     label = self.dataset.targets[index]
#                 else:
#                     label = self.dataset.target_transform(self.dataset.targets[index])
#             else:
#                 label = self.dataset[index][1]
#             if label in sub_labels:
#                 self.sub_indeces.append(index)
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.sub_indeces)

#     def __getitem__(self, index):
#         sample = self.dataset[self.sub_indeces[index]]
#         if self.target_transform:
#             target = self.target_transform(sample[1])
#             sample = (sample[0], target)
#         return sample




from torch.utils.data import Dataset


class SubDataset(Dataset):
    """
    Sub-sample a dataset based on subclass labels (y_sub).

    Supports datasets that return:
      (x, y)
      (x, y_sub, y_main)
    """

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        self.target_transform = target_transform

        for index in range(len(self.dataset)):
            sample = self.dataset[index]

            # ---- flat dataset (x, y)
            if len(sample) == 2:
                _, y = sample
                y_sub = y

            # ---- hierarchical dataset (x, y_sub, y_main)
            elif len(sample) == 3:
                _, y_sub, _ = sample

            else:
                raise ValueError(
                    f"Unsupported sample format with length {len(sample)}"
                )

            if y_sub in sub_labels:
                self.sub_indeces.append(index)

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]

        # If we want to remap subclass labels (rare, but keep compatibility)
        if self.target_transform:
            if len(sample) == 2:
                x, y = sample
                return x, self.target_transform(y)
            else:
                x, y_sub, y_main = sample
                return x, self.target_transform(y_sub), y_main

        return sample
