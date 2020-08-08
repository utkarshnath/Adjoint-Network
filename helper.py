from fastai import datasets
from fastai.vision import *
import pickle, gzip, math, torch, matplotlib as mpl
from torch import tensor, nn
from pathlib import Path
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import DataLoader, Dataset
from run import DataBunch
from datablock import *
from torchvision import transforms

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'
CIFAR10_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz'
class MNISTDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return len(self.x)

def get_data():
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
    # MNIST
    x_train, y_train, x_valid, y_valid = get_data()
    
    train_mean, train_std = get_stats(x_train)
    x_train = normalize(x_train, train_mean, train_std)
    x_valid = normalize(x_valid, train_mean, train_std)

    train_ds = MNISTDataset(x_train[:,:], y_train)
    valid_ds = MNISTDataset(x_valid[:,:], y_valid)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, shuffle=False)

    data = DataBunch(train_dl, valid_dl)

    return data


def load_cifar_data(batch_size, image_size,size):
    if size==10:
       path = datasets.untar_data(URLs.CIFAR)
    else:
       path = datasets.untar_data(URLs.CIFAR_100)
    stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))
    
    tfms = (get_transforms(do_flip=True,flip_vert=False,max_rotate=25))
    data = ImageDataBunch.from_folder(path, valid='test', size=image_size,ds_tfms=tfms,bs = batch_size)
    data.normalize(imagenet_stats)

    print("Loaded data")
    return data


def load_data(batch_size, image_size, dataset=1):
    if dataset==0:
        path = Path('/scratch/work/public/imagenet/')
    elif dataset==1:
        path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)
    elif dataset==2:
        path = datasets.untar_data(datasets.URLs.IMAGEWOOF_160)
    
    train_transforms = [make_rgb, RandomResizedCrop(image_size, scale=(0.35,1)), PilRandomFlip(), np_to_float]
    valid_transforms = [make_rgb, CenterCrop(image_size), np_to_float]    

    data = Data(path, batch_size=batch_size, image_transforms=train_transforms, valid_image_transforms=valid_transforms,num_workers=8)
    
    print("Loaded data")
    return data

def load_fastai_data(batch_size, image_size):
    path = untar_data(URLs.PETS)
    path_img = path/'images'
    fnames = get_image_files(path_img)
    pat = r'/([^/]+)_\d+.jpg$'
    data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=image_size, bs=batch_size).normalize(imagenet_stats)

    print("Loaded data")
    return data
