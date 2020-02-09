from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
from torch import tensor, nn
from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl

import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import DataLoader, Dataset

from run import DataBunch

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'

class MNISTDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return len(self.x)

def get_data():
    # path = datasets.download_data(MNIST_URL, ext='.gz')
    path = 'mnist.pkl.gz'
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s): 
    return (x-m)/s

def get_stats(x):
    mean, std = x.mean(), x.std()
    return mean, std

def get_data_bunch(batch_size):
    x_train, y_train, x_valid, y_valid = get_data()
    print(x_train.size())
    train_mean, train_std = get_stats(x_train)
    x_train = normalize(x_train, train_mean, train_std)
    x_valid = normalize(x_valid, train_mean, train_std)

    train_ds = MNISTDataset(x_train, y_train)
    valid_ds = MNISTDataset(x_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, shuffle=False)

    data = DataBunch(train_dl, valid_dl)

    return data
