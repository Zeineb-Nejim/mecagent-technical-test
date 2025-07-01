import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
import random

class CadQueryDataset(Dataset):
    def __init__(self, ds, max_length=128):
        self.samples = ds
        self.max_length = max_length
        self.chars = sorted(list({c for s in ds for c in s['cadquery']}))
        self.char2idx = {c: i+1 for i, c in enumerate(self.chars)}  # 0 is padding
        self.idx2char = {i+1: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 1

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = self.transform(sample['image'])
        code = sample['cadquery'][:self.max_length]
        code_idx = [self.char2idx[c] for c in code]
        code_idx += [0] * (self.max_length - len(code_idx))  # pad
        return img, torch.tensor(code_idx), code

def split_dataset(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits the dataset into train, validation, and test sets.
    Args:
        samples (list): List of dataset samples.
        train_ratio (float): Proportion for training set.
        val_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for test set.
        seed (int): Random seed for reproducibility.
    Returns:
        train_samples, val_samples, test_samples (lists)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    n = len(samples)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, val_samples, test_samples

# Example usage:
# train_dataset = CadQueryDataset(ds[0])
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)