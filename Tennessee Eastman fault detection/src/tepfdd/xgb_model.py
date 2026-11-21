"""Windowed features for the gradient-boosted classifier (causal: a window ending at sample t uses samples t-W+1..t)."""
import numpy as np


def window_features(X, W):
    """X (R, T, p) -> (R, T-W+1, 4p) float32: window mean, std, last value, last-minus-first (trend)."""
    X = np.asarray(X, np.float64)
    R, T, p = X.shape
    c1 = np.concatenate([np.zeros((R, 1, p)), np.cumsum(X, 1)], 1)
    c2 = np.concatenate([np.zeros((R, 1, p)), np.cumsum(X * X, 1)], 1)
    s1 = c1[:, W:] - c1[:, :-W]
    s2 = c2[:, W:] - c2[:, :-W]
    mean = s1 / W
    std = np.sqrt(np.maximum(s2 / W - mean ** 2, 0))
    last = X[:, W - 1:]
    first = X[:, :T - W + 1]
    return np.concatenate([mean, std, last, last - first], 2).astype(np.float32)
