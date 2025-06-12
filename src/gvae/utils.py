import torch
from torch.nn.functional import softplus
from torch import inf
import torch
from pandas import Timestamp
from datetime import timedelta
import numpy as np
import os, json, sys, argparse

def to_positive(xtilde): return torch.nn.functional.softplus(xtilde, beta=1, threshold=5)

def from_positive(x): return x + torch.log(-torch.expm1(-x))

def to_sigma(sigmatilde): return to_positive(sigmatilde)

def from_sigma(sigma): return from_positive(sigma)

def matrix_normalizer(matrix):
    return matrix / torch.linalg.norm(matrix, dim=0, ord=2, keepdim=True)

def log_prob(dist="Normal", params=None, targets=None):
    if dist=="Normal":
        return -0.5*torch.log(2*torch.tensor(torch.pi)) - torch.log(params["sigma"]) - 0.5*((targets-params["mu"])/params["sigma"])**2
    elif dist=="MultivariateNormal":
        ...
    else:
        raise ValueError("Unknown distribution.")

def kl_divergence(dist="Normal", params=None, prior_params=None):
    if dist=="Normal":
        var_ratio = (params["sigma"] / prior_params["sigma"]).pow(2)
        t1 = ((params["mu"] - prior_params["mu"]) / prior_params["sigma"]).pow(2)
        return 1.5 * (var_ratio + t1 - 1 - var_ratio.log())
    else:
        raise ValueError("Unknown distribution.")

def zero_preserved_log_stats(X):
    Y = np.copy(X)
    is_zero = (Y == 0)
    Y[is_zero] = np.nan
    Y_log = np.log(Y)
    nonzero_mean = np.nanmean(Y_log, axis=0, keepdims=True)
    nonzero_std = np.nanstd(Y_log, axis=0, keepdims=True)
    return nonzero_mean, nonzero_std

def zero_preserved_log_normalize(X, nonzero_mean, nonzero_std, log_output=False, zero_id=-3, shift=1.0):
    Y = np.copy(X)
    is_zero = (Y == 0)
    Y[is_zero] = np.nan
    Y_log = np.log(Y)
    Y_log = (Y_log-nonzero_mean)/nonzero_std + shift
    if log_output: Y = Y_log
    else: Y = np.exp(Y_log)
    Y[is_zero] = zero_id
    return Y

def zero_preserved_log_denormalize(Y, nonzero_mean, nonzero_std, log_input=False, zero_id=-3, shift=1.0):
    X = np.copy(Y)
    is_zero = (X == zero_id)
    X[is_zero] = np.nan
    if log_input: X_log = X
    else: X_log = np.log(X)
    X_log = (X_log-shift)*nonzero_std + nonzero_mean
    X = np.exp(X_log)
    X[is_zero] = 0
    return X

def two_sided_log_transform(X, alpha=1): return (np.maximum(0,X)*np.log(1+np.maximum(0,X)/np.abs(X+1e-24)*X/alpha) - np.maximum(0,-X)*np.log(1-np.maximum(0,-X)/np.abs(X+1e-24)*X/alpha)) / np.abs(X+1e-24) * alpha

def two_sided_log_transform_inverse(Y, alpha=1): return (np.maximum(0,Y)*(np.exp(np.maximum(0,Y)/np.abs(Y+1e-24)*Y/alpha)-1) - np.maximum(0,-Y)*(np.exp(-np.maximum(0,-Y)/np.abs(Y+1e-24)*Y/alpha)-1)) / np.abs(Y+1e-24) * alpha

def two_sided_log_normalize(X, mean, std, alpha=1):
    Y_2log = two_sided_log_transform(X, alpha)
    Y = (Y_2log - mean) / std
    return Y

def two_sided_log_denormalize(Y, mean, std, alpha=1):
    Y_2log = Y * std + mean
    X = two_sided_log_transform_inverse(Y_2log, alpha)
    return X

class IdentityTransformer():
    def fit_transform(self, data): return data
    def transform(self, data): return data
    def inverse_transform(self, data): return data

class CircularTransformer():
    def __init__(self, max_conds=None, min_conds=None):
        self.max_conds, self.min_conds = max_conds, min_conds

    def fit_transform(self, data):
        if self.max_conds is None: self.max_conds = np.max(data,axis=0,keepdims=False)
        if self.min_conds is None: self.min_conds = np.min(data,axis=0,keepdims=False)
        angles = (data-self.min_conds)/(self.max_conds-self.min_conds+1)*2*np.pi
        return np.concatenate((np.cos(angles),np.sin(angles)),axis=1)
    
    def transform(self, data):
        angles = (data-self.min_conds)/(self.max_conds-self.min_conds+1)*2*np.pi
        return np.concatenate((np.cos(angles),np.sin(angles)),axis=1)

    def inverse_transform(self, data):
        return (np.arctan2(data[:,1],data[:,0])/(2*np.pi)*(self.max_conds-self.min_conds+1)+self.min_conds)
    
class DirichletTransformer():
    def __init__(self, gammas=None, transform_style="sample"):
        self.transform_style = transform_style
        self.gamma_min = gammas.min(axis=0)
        self.gamma_max = gammas.max(axis=0)
        self.total_gamma_max = gammas.sum(axis=1).max()
        self.total_gamma_min = gammas.sum(axis=1).min()

    def transform(self, gammas, transform_style=None, num_samples=1):
        if transform_style is not None: self.transform_style = transform_style
        else: transform_style = self.transform_style

        if self.transform_style == "sample": 
            gamma_rvs = np.random.gamma(gammas, 1, size=(num_samples, gammas.shape[0], gammas.shape[1]))
            return (gamma_rvs/gamma_rvs.sum(-1, keepdims=True))
        elif self.transform_style == "embed": 
            embedding = np.zeros((gammas.shape[0], gammas.shape[1]+1))
            embedding[:,:-1] = gammas/gammas.sum(axis=1, keepdims=True)
            embedding[:,-1] = (gammas.sum(axis=1)-self.total_gamma_min)/(self.total_gamma_max-self.total_gamma_min+1e-6)
            return embedding
        elif self.transform_style == "mean":
            return gammas/gammas.sum(axis=1, keepdims=True)
        elif self.transform_style == "scaled":
            return (gammas-self.gamma_min)/(self.gamma_max-self.gamma_min+1e-6)
        else:
            raise NotImplementedError
        
class MinMaxTransformer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X):
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        X_scaled = X * self.scale_ + self.min_
        X_clipped = np.clip(X_scaled, self.feature_range[0], self.feature_range[1])
        return X_clipped

    def inverse_transform(self, X):
        X_inversed = (X - self.min_) / self.scale_
        X_inversed_clipped = np.clip(X_inversed, self.data_min_, self.data_max_)
        return X_inversed_clipped

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0.1):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, score, pbar=None): 
        if self.best_score is None: self.best_score = score

        elif score <= (self.best_score + self.delta) and not self.early_stop:
            self.counter += 1
            if pbar is not None: pbar.write(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: 
                self.early_stop = True
                if pbar is not None: pbar.write("Early stopping initiated.")
                else: print("Early stopping initiated.")

        elif score > (self.best_score + self.delta) and not self.early_stop:
            self.best_score = score
            self.counter = 0
            if pbar is not None: pbar.write(f"New (significant) best score: {self.best_score:5e}")
            else: print(f"New (significant) best score: {self.best_score:5e}")

def find_matching_model(base_dir, user_config_dict):
    for subdir, _, _ in os.walk(base_dir):
        model_path = os.path.join(subdir, 'user_config_dict.json')
        if os.path.exists(model_path):
            with open(model_path, 'r') as f: saved_user_config_dict = json.load(f)
            if saved_user_config_dict==user_config_dict: return subdir
    return None

def get_latest_path(base_dir):
    paths = [os.path.join(base_dir,f) for f in os.listdir(base_dir)]
    latest_path = paths[np.argmax([os.path.getmtime(f) for f in paths])]
    return latest_path

def flatten_dict(nested_dict, parent_key='', sep='_'):
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, json.dumps(v)))  # Convert list to JSON string
        else:
            items.append((new_key, v))
    return dict(items)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def lower_band(M, k):
    P = np.zeros((k+1, M.shape[0]))
    for i in range(k+1): P[i, i:M.shape[0]] = np.diag(M, k=-i)
    return P[1:]

def full_band(M, k):
    m = M.shape[0]
    band_width = 2 * k + 1
    P = np.zeros((band_width, m))
    for i in range(-k, k + 1): P[k + i, max(0, i):m + min(0, i)] = np.diag(M, k=i)
    return P
