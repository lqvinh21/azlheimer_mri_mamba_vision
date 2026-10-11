import numpy as np
import torch
import torch.nn as nn


def make_weighted_cross_entropy(train_dataset, num_classes, device):
    labels = [label for _, label in train_dataset.samples]
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights)
