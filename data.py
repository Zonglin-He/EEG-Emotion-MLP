import os
import torch
import numpy as np

from sklearn.cluster import KMeans
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy.interpolate import interp1d

from scipy.signal import find_peaks
import pandas as pd
import pickle
from scipy.signal import butter, filtfilt, find_peaks
from torch.utils.data import Dataset, DataLoader
import numpy as np


# data_dir = 'Data'
# data = pd.read_csv(os.path.join(data_dir, 'hr.csv'))
#
# id = data['id'].drop_duplicates()
# id.to_csv(os.path.join(data_dir, 'id_list.csv'), index=False)


class DEAPDataset(Dataset):
    """
    Dataset class for DEAP EEG data.

    Args:
        args: Arguments containing data directories
        subject_files (list): List of subject files to load
        norm_params (dict): Dictionary containing normalization parameters.
                           If None, calculate from data (training mode)
    """

    def __init__(self, data):
        """
        Args:
            args: Arguments containing data directories
            subject_files: List of subject files to load
            norm_params: Dictionary containing normalization parameters.
                       If None, calculate from data (training mode)
        """
        super(DEAPDataset, self).__init__()
        tt = [f't_{t}' for t in range(29)]
        self.data = data[tt].values
        self.labels = data[['Valence', 'Arousal']].values

    def __getitem__(self, index):
        """
        Get a single data item.

        Args:
            index (int): Index of the item

        Returns:
            tuple: (data, features) containing EEG data and features
        """
        x = self.data[index]
        x = torch.from_numpy(x).float()
        y = self.labels[index]
        y = torch.from_numpy(y).float()
        y = ((y - 1) / (9 - 1)) * 2 - 1

        return x, y

    def __len__(self):
        """
        Get dataset length.

        Returns:
            int: Number of items in dataset
        """
        return self.data.shape[0]


def data_loader(valid_subject):
    """
    Create data loaders for training and validation.

    Args:
        args: Arguments containing batch size and directories
        valid_subject (str): Subject ID for validation

    Returns:
        tuple: (train_loader, valid_loader)

    """

    data_dir = 'Data'
    data = pd.read_csv(os.path.join(data_dir, 'hr.csv'))

    # id = data['id']
    # id.to_csv(os.path.join(data_dir, 'id_list.csv'), index=False)

    train_data = data[data['id'] != valid_subject]
    valid_data = data[data['id'] == valid_subject]

    # Create training dataset and get normalization parameters
    train_dataset = DEAPDataset(train_data)
    valid_dataset = DEAPDataset(valid_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
    )

    return train_loader, valid_loader
