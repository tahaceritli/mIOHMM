import numpy as np
import pickle
import torch


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath):
    with open(filepath, "rb") as handle:
        data = pickle.load(handle)
    return data


def normalize(A, axis=None):
    Z = torch.sum(A, axis=axis, keepdims=True)
    idx = np.where(Z == 0)
    Z[idx] = 1
    return A / Z


def normalize_exp(log_P, axis=None):
    a, _ = torch.max(log_P, keepdims=True, axis=axis)
    P = normalize(torch.exp(log_P - a), axis=axis)
    return P
