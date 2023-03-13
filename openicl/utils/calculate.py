import numpy as np


def entropy(probs: np.array, label_dim: int = 0, mask=None):
    if mask is None:
        return - (probs * np.log(probs)).sum(label_dim)
    return - (mask * probs * np.log(probs)).sum(label_dim)
