import PIL, os, mimetypes
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())

import random
from random import randint
from collections import Iterable
import math
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader, Dataset

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

class ListManager():
    def __init__(self, items): self.items = items
    def __getitem__(self, idx):
        try: return self.items[idx]
        except TypeError:
            if isinstance(idx[0],bool):
                assert len(idx)==len(self) # bool mask
                return [o for m,o in zip(idx,self.items) if m]
            return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res

def get_files(path, extensions):
    path = Path(path)

    files = []
    for p, ds, fs in os.walk(path):
        files += _get_files(p, fs, extensions)

    return files


def _get_files(p, fs, extensions):
    p = Path(p)
    return [p/f for f in fs if not f.startswith('.') and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]


def compose(x, funcs, *args, **kwargs):
    if not funcs: return x
    for f in funcs: x = f(x, **kwargs)
    return x


class ImageList(ListManager):
    def __init__(self, path, transforms=None):
        image_extensions = set(k.lower() for k,v in mimetypes.types_map.items() if v.startswith('image/'))
        self.items = get_files(path, image_extensions)
        self.transforms = transforms

    def get(self, path): return PIL.Image.open(path)
    def _get(self, i): return compose(self.get(i), self.transforms)
    def __getitem__(self, i):
        idxs = super().__getitem__(i)
        if isinstance(idxs, list): return [self._get(v) for v in idxs]
        return self._get(idxs)


def path_to_label_vocab(path): return path.parent.name
def create_label_vocab(image_list):
    labels_vocab = {}
    for path in image_list:
        name = path_to_label_vocab(path)
        if name not in labels_vocab:
            labels_vocab[name] = len(labels_vocab)
    return labels_vocab

class CuratedDataset(Dataset):
    def __init__(self, image_list, vocab=None):
        self.x = image_list
        self.y = [vocab[path_to_label_vocab(p)] for p in image_list.items]

    def __getitem__(self, i): return self.x[i], self.y[i]
    def __len__(self): return len(self.x)

    def visualize(self, idx):
        file_path = self.x.items[idx]
        image = self.x.get(file_path)

        return image, file_path

    def check_labels(self, number=10):
        print(self.__class__.__name__)
        for i in range(number):
            idx = random.randrange(len(self))

            label = self.y[idx]
            file_path = self.x.items[idx]

            print(file_path, label)


class Data():
    def __init__(self, path, batch_size=128, image_transforms=None, valid_image_transforms=None, num_workers=4):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_image_transforms = image_transforms
        if valid_image_transforms is None: valid_image_transforms = image_transforms

        train_list = ImageList(self.train_path, train_image_transforms)
        valid_list = ImageList(self.valid_path, valid_image_transforms)

        vocab = create_label_vocab(train_list)

        self.train_ds = CuratedDataset(train_list, vocab)
        self.valid_ds = CuratedDataset(valid_list, vocab)

    @property
    def train_path(self): return self.path/'train'
    @property
    def valid_path(self): return self.path/'val'


    #@property
    #def train_path(self): return '/scratch/work/public/imagenet/train'
    #@property
    #def valid_path(self): return '/scratch/un270/val' 

    @property
    def train_dl(self): return DataLoader(self.train_ds, self.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=self.num_workers)
    @property
    def valid_dl(self): return DataLoader(self.valid_ds, 128, shuffle=False, drop_last=True, num_workers=self.num_workers)


def make_rgb(item): return item.convert('RGB')

class ResizeFixed():
    def __init__(self,size):
        if isinstance(size,int): size=(size,size)
        self.size = size

    def __call__(self, item): return item.resize(self.size, PIL.Image.BILINEAR)

def to_byte_tensor(item):
    res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
    w,h = item.size
    return res.view(h,w,-1).permute(2,0,1)
def to_float_tensor(item): return item.float().div_(255.)
def np_to_float(x): return torch.from_numpy(np.array(x, dtype=np.float32, copy=False)).permute(2,0,1).contiguous()/255.


class PilRandomFlip():
    def __init__(self, p=0.5): self.p=p
    def __call__(self, x):
        return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random()<self.p else x


def process_sz(sz):
    sz = listify(sz)
    return tuple(sz if len(sz)==2 else [sz[0],sz[0]])

def default_crop_size(w,h): return [w,w] if w < h else [h,h]

class GeneralCrop():
    def __init__(self, size, crop_size=None, resample=PIL.Image.BILINEAR):
        self.resample,self.size = resample,process_sz(size)
        self.crop_size = None if crop_size is None else process_sz(crop_size)

    def default_crop_size(self, w,h): return default_crop_size(w,h)

    def __call__(self, x):
        csize = self.default_crop_size(*x.size) if self.crop_size is None else self.crop_size
        return x.transform(self.size, PIL.Image.EXTENT, self.get_corners(*x.size, *csize), resample=self.resample)

    def get_corners(self, w, h): return (0,0,w,h)

class CenterCrop(GeneralCrop):
    def __init__(self, size, scale=1.14, resample=PIL.Image.BILINEAR):
        super().__init__(size, resample=resample)
        self.scale = scale

    def default_crop_size(self, w,h): return [w/self.scale,h/self.scale]

    def get_corners(self, w, h, wc, hc):
        return ((w-wc)//2, (h-hc)//2, (w-wc)//2+wc, (h-hc)//2+hc)

class RandomResizedCrop(GeneralCrop):
    def __init__(self, size, scale=(0.08,1.0), ratio=(3./4., 4./3.), resample=PIL.Image.BILINEAR):
        super().__init__(size, resample=resample)
        self.scale,self.ratio = scale,ratio

    def get_corners(self, w, h, wc, hc):
        area = w*h
        #Tries 5 times to get a proper crop inside the image.
        for attempt in range(5):
            area = random.uniform(*self.scale) * area
            ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            new_w = int(round(math.sqrt(area * ratio)))
            new_h = int(round(math.sqrt(area / ratio)))
            if new_w <= w and new_h <= h:
                left = random.randint(0, w - new_w)
                top  = random.randint(0, h - new_h)
                return (left, top, left + new_w, top + new_h)

        # Fallback to squish
        if   w/h < self.ratio[0]: size = (w, int(w/self.ratio[0]))
        elif w/h > self.ratio[1]: size = (int(h*self.ratio[1]), h)
        else:                     size = (w, h)
        return ((w-size[0])//2, (h-size[1])//2, (w+size[0])//2, (h+size[1])//2)


