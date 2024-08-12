# Refactored dataset-related functions outside of notebook 
# for easier reusibility + readibility
# @Author Jack Bosco

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

def toSequential(idx, full_list, timeStep=24, gap=8):
    """
    This function transforms given data into sequential samples with specified time steps and gaps.

    Parameters:
    idx (int): Index of the specific dataset in 'full_list' to be processed.
    full_list (list): A list of datasets (each being a numpy array).
    timeStep (int): Number of time steps in each sequential sample.
    gap (int): Gap between start points of successive sequential samples.

    Returns:
    stockSeq (numpy.ndarray): Normalized sequential data samples.
    labelSeq (numpy.ndarray): Normalized closing prices for each time step in the sequence.
    diffSeq (numpy.ndarray): Normalized differences of closing prices between successive steps.
    realDiffSeq (numpy.ndarray): Real differences of closing prices between successive steps (not normalized).
    """
    
    # Extract the closing prices from the dataset corresponding to the provided index
    closing = full_list[idx][:, 3]
    
    # Extract all data points except for the last one for processing
    data = full_list[idx][:-1]
    
    # Calculate the length of the available data
    data_length = len(data)
    
    # Calculate the number of sequential samples that can be created
    count = (data_length - timeStep) // gap + 1
    
    # Initialize lists to store the results
    stockSeq = []
    labelSeq = []
    diffSeq = []
    realDiffSeq = []
    
    for i in range(count):
        # Extract the segment of data for the current time step
        segData = data[gap * i : gap * i + timeStep]
        
        # Extract the corresponding closing prices for the current time step segment + 1 for label
        segClosing = closing[gap * i : gap * i + timeStep + 1]

        # Normalize the segment data by subtracting its mean and dividing by its standard deviation
        segDataNorm = np.nan_to_num((segData - segData.mean(axis=0, keepdims=True)) / segData.std(axis=0, keepdims=True))
        
        # Normalize the segment closing prices similarly
        segClosingNorm = (segClosing - segClosing.mean()) / segClosing.std()
        
        # Append the normalized segment data to the stock sequence list
        stockSeq.append(segDataNorm)
        labelSeq.append(segClosingNorm[1:])
        
        # Append the normalized differences between successive closing prices to the difference sequence list
        diffSeq.append(segClosingNorm[1:] - segClosingNorm[:-1])
        
        # Append the actual differences between successive closing prices to the real difference sequence list (not normalized)
        realDiffSeq.append(segClosing[1:] - segClosing[:-1])
    
    # Transform the lists into numpy arrays for efficient computation
    stockSeq = np.array(stockSeq)
    labelSeq = np.array(labelSeq)
    diffSeq = np.array(diffSeq)
    realDiffSeq = np.array(realDiffSeq)
    
    # Return the sequences as numpy arrays with 'float32' data type for optimization
    return stockSeq.astype('float32'), labelSeq.astype('float32'), diffSeq.astype('float32'), realDiffSeq.astype('float32')

class StockDataset(Dataset):
    """Custom PyTorch Dataset class for loading and processing stock data."""
    
    def __init__(self, id_list, full_list, transform=None, timestep=24, gap=8):
        """
        Initialize the StockDataset object.
        
        Args:
            id_list (list): List of stock identifiers to be included in the dataset.
            full_list (list): Full list of stock data from which to extract the sequences.
            transform (callable, optional): Optional transform to be applied on a sample.
            timestep (int, optional): Number of timesteps for each sequence. Default is 24.
            gap (int, optional): Gap between sequences. Default is 8.
        """
        self.transform = transform
        self.id_list = id_list
        
        stock_cohort = []
        closing_cohort = []
        diff_cohort = []
        real_diff_cohort = []
        
        # Load data into cohorts for each stock ID
        for i in self.id_list:
            X, y, z, zp = toSequential(i, full_list, timeStep=timestep, gap=gap)
            stock_cohort.append(X)
            closing_cohort.append(y)
            diff_cohort.append(z)
            real_diff_cohort.append(zp)
        
        # Concatenate lists into numpy arrays for entire dataset
        self.X = np.concatenate(stock_cohort, axis=0)
        self.y = np.concatenate(closing_cohort, axis=0)
        self.z = np.concatenate(diff_cohort, axis=0)
        self.zp = np.concatenate(real_diff_cohort, axis=0)
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.
        
        Args:
            idx (int or tensor): Index of the sample to retrieve.
        
        Returns:
            tuple: A tuple containing data and labels (data, label1, label2, label3).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.X[idx]
        label1 = self.y[idx]
        label2 = self.z[idx]
        label3 = self.zp[idx]
        
        # Apply the optional transform, if any
        if self.transform:
            data = self.transform(data)
        
        return (data, label1, label2, label3)
    
    def getDS(self):
        """
        Returns the entire dataset as arrays.
        
        Returns:
            tuple: A tuple containing all data arrays (X, y, z, zp).
        """
        return self.X, self.y, self.z, self.zp