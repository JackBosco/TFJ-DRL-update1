# Refactored dataset-related functions outside of notebook 
# for easier reusibility + readibility
# @Author Jack Bosco

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from .technical_analysis import toSequential

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