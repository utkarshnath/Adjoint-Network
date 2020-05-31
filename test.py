from fastai import datasets
from fastai.vision import *
import pickle, gzip, math, torch, matplotlib as mpl
from torch import tensor, nn
from pathlib import Path
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import DataLoader, Dataset
#from datablock import make_rgb, np_to_float, to_byte_tensor, to_float_tensor
#from datablock import Data, ResizeFixed, RandomResizedCrop, CenterCrop, PilRandomFlip
from run import DataBunch
from datablock import *
from helper import *

def copy_files_make_csv(t, p, csv_name):
    if t not in ['train', 'test', 'valid']:
        raise ValueError("Argument must be either 'train' or 'test'")
    labels_path = p / (t + '.txt')
    fastai_labels = p / csv_name
    
    with open(fastai_labels, 'a') as labels:
        with open(labels_path) as f:
            for line in f:
                    jpg_filename, category = line.strip('\n').split(' ')
                    jpg = jpg_filename[:4]
                    filename = jpg_filename[4:]
                    target_path = str(p) + '/' + t + '/' + filename

                    if not os.path.exists(os.path.dirname(target_path)):
                        os.makedirs(os.path.dirname(target_path))
                        print(f'Created new dir: {os.path.dirname(target_path)}')

                    src = str(p/jpg/filename)
                    dst = str(p/target_path)
                    
#                     print(src + ' -> ' + dst)

                    shutil.copy2(src, dst)
                    labels.write(t + '/' + filename + ',' + category + '\n')
                
    print(labels_path)

def prepare_data(path):
    labels_path = path/'labels.csv'
    labels_test_path = path/'labels_test.csv'
    train_path = path/'train'
    test_path = path/'test'
    
    # Clean up after last run
    if labels_path.exists():
        os.remove(str(labels_path))
        print("Removed existing labels.csv")
    
    if labels_test_path.exists():
        os.remove(str(labels_test_path))
        print("Removed existing labels_test.csv")
        
    if train_path.exists():
        shutil.rmtree(path/'train')
        print("Removed existing train folder.")
        
    if test_path.exists():    
        shutil.rmtree(path/'test')
        print("Removed existing test folder.")
        
    # Do actual preparation
    for t in ['train','valid']:
        copy_files_make_csv(t, path, 'labels.csv')
        
    # Prepare the test folder
    copy_files_make_csv('test', path, 'labels_test.csv')

path = Path('/home/un270/.fastai/data/oxford-102-flowers')
prepare_data(path)

bs=64
tfms = get_transforms()
data = ImageDataBunch.from_csv(path, size=224, ds_tfms=tfms, bs=bs).normalize(imagenet_stats)

