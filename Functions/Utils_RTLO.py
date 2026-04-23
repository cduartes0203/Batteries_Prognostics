
import numpy as np
import pandas as pd
import scipy.io
import pickle

from scipy import signal
from scipy.stats import entropy, kurtosis, entropy
from scipy.signal import hilbert, chirp
import os
import pickle as pkl
import re
import scipy.stats as stats
import math
from scipy import signal
from scipy.signal import savgol_filter, stft
from scipy.ndimage import gaussian_filter1d, median_filter
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE

def ResizeSignal(data, m):
    """
    Interpola um array de tamanho n para o tamanho m (m > n).
    """
    n = len(data)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, m)
    
    return np.interp(x_new, x_old, data)

def PrepareDataPast(sig, n, m):
    X, Y = [], []
    s = n - m + 1 
    for i in range(len(sig) - n - 1):
        X.append(sig[i : i + n])
        Y.append(sig[i + s : i + s + m])
    return np.array(X)[:-1], np.array(Y)[:-1]

def PrepareDataAhead(sig, n, m):
    X, Y = [], []
    s = n - m + 1 
    for i in range(len(sig)-n-m+1):
        X.append(sig[i : i + n])
        Y.append(sig[i + n : i +n+ m])
    return np.array(X)[:-1], np.array(Y)[:-1]

def PrepareData(sig,n,m,mode='past'):
    if mode == 'past':
        return PrepareDataPast(sig, n, m)
    elif mode == 'ahead':
        return PrepareDataAhead(sig, n, m)


def XavierUniform(shape,sd):
    np.random.seed(sd)
    n_in, n_out = shape
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=shape) 

def Activation(x,mode='tanh'):
    '''mode: tanh, sigmoid, relu'''

    if mode == 'tanh':
        return np.tanh(x)
    elif mode == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif mode == 'relu':
        return np.maximum(0, x)

def dActivation(x,mode='tanh'):
    '''mode: tanh, sigmoid, relu'''

    if mode == 'tanh':
        return 1 - np.tanh(x)**2
    elif mode == 'sigmoid':
        return (1 / (1 + np.exp(-x))) * (1- (1 / (1 + np.exp(-x))))
    elif mode == 'relu':
        return np.where(x > 0, 1, 0)