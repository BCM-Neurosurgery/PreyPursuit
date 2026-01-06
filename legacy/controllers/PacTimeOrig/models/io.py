import numpy as np
import pandas as pd
import scipy
from scipy.integrate import cumtrapz
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence


#### TENSORFLOW CODE

class VariableLengthDataGenerator(Sequence):
    def __init__(self, X_data, y_data, batch_size):
        """
        Initializes the data generator.
        X_data: list of numpy arrays with different lengths (number of timesteps)
        y_data: list of numpy arrays or values corresponding to each X_data sequence
        batch_size: number of sequences per batch
        """
        self.X_data = X_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.indices = np.arange(len(X_data))

    def __len__(self):
        # Returns the number of batches per epoch
        return int(np.ceil(len(self.X_data) / self.batch_size))

    def __getitem__(self, index):
        # Get the indices for the current batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Get the batch data
        X_batch = [self.X_data[i] for i in batch_indices]
        y_batch = [self.y_data[i] for i in batch_indices]

        # Pad the sequences in the batch to the same length
        X_batch_padded = pad_sequences(X_batch, padding='post', dtype='float32')
        y_batch_padded = pad_sequences(y_batch, padding='post', dtype='float32')

        return X_batch_padded, y_batch_padded

    def on_epoch_end(self):
        # Optionally shuffle the data at the end of each epoch
        np.random.shuffle(self.indices)


#### TORCH CODE

class VariableLengthDataset(Dataset):
    def __init__(self, X_data, y_data):
        """
        Initializes the dataset that is variable length time-series and pads them with zeros at the end
        X_data: list of numpy arrays with different lengths (number of timesteps)
        y_data: list of numpy arrays or values corresponding to each X_data sequence
        """
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        # Returns the total number of samples
        return len(self.X_data)

    def __getitem__(self, idx):
        # Returns one data pair (input and target)
        return torch.tensor(self.X_data[idx], dtype=torch.float32), torch.tensor(self.y_data[idx], dtype=torch.float32)

def torchloader(dataset, batch_size=None):
    if batch_size is None:
        batch_size = 32
    # DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return dataloader


def collate_fn(batch):
    X_batch, y_batch = zip(*batch)
    X_batch = [torch.tensor(seq, dtype=torch.float32) for seq in X_batch]
    y_batch = [torch.tensor(seq, dtype=torch.float32) for seq in y_batch]
    seq_lengths = torch.tensor([len(seq) for seq in X_batch])

    X_batch_padded = pad_sequence(X_batch, batch_first=True, padding_value=0.0)
    y_batch_padded = pad_sequence(y_batch, batch_first=True, padding_value=0.0)

    return X_batch_padded, y_batch_padded, seq_lengths


#Controller variants

def collate_fn_control(batch):
    """
    Custom collate function to handle variable-length sequences and initial positions.
    batch: list of tuples (X_seq, y_seq, initial_position)
    """

    # Unpack the batch
    X_seqs, y_seqs, initial_positions = zip(*batch)

    # Get sequence lengths
    seq_lengths = [len(seq) for seq in X_seqs]

    # Pad sequences
    X_padded = pad_sequence(X_seqs, batch_first=True)
    y_padded = pad_sequence(y_seqs, batch_first=True)

    # Stack initial positions
    initial_positions = torch.stack(initial_positions, dim=0)

    # Convert seq_lengths to tensor
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)

    return X_padded, y_padded, initial_positions, seq_lengths


def torchloaderControl(dataset, batch_size=None):
    if batch_size is None:
        batch_size = 32
    # DataLoader for batching with custom collate_fn
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_control, shuffle=True)
    return dataloader


class VariableLengthDatasetControl(Dataset):
    def __init__(self, X_data, y_data):
        """
        Initializes the dataset with variable-length time-series data.
        X_data: list of numpy arrays with different lengths (number of timesteps)
        y_data: list of numpy arrays corresponding to each X_data sequence
        """
        self.X_data = X_data  # List of numpy arrays
        self.y_data = y_data  # List of numpy arrays

    def __len__(self):
        # Returns the total number of samples
        return len(self.X_data)

    def __getitem__(self, idx):
        # Get the sequence data
        X_seq = self.X_data[idx]
        y_seq = self.y_data[idx]

        # Extract initial position from specified columns (e.g., columns 0 and 2)
        initial_position = X_seq[0, [0, 2]]  # Shape: (2,)

        # Optionally remove columns 0 and 2 from X_seq if they are not needed anymore
        X_seq = np.delete(X_seq, [0, 2], axis=1)

        # Convert to tensors
        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        y_seq = torch.tensor(y_seq, dtype=torch.float32)
        initial_position = torch.tensor(initial_position, dtype=torch.float32)

        return X_seq, y_seq, initial_position