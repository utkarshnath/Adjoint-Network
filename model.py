import torch.nn as nn
import torch.nn.init as init
from myconv import myconv2d
from mask import *

randommask = 1

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def mnist_resize(x): return x.view(-1, 1, 28, 28)
def cifar_resize(x): return x.view(-1, 3, 32, 32)
def imagenet_resize(x): return x.view(-1, 3, 128, 128)
def noop(x): return x

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

act_func = nn.ReLU()

first = True
def conv(ni, no, ks, s=1, bias=False,mask_layer=False):
    return nn.Conv2d(ni, no, kernel_size=ks, stride=s, padding=ks//2, bias=bias)

def conv_layer(ni, no, ks, s, zero_bn=False, act=True,mask_layer=True):
    bn = nn.BatchNorm2d(no)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, no, ks=ks, s=s,mask_layer=mask_layer), bn]
    if act: layers += [act_func]
    return nn.Sequential(*layers)

def init_cnn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)): init.kaiming_normal_(m.weight, a=1.0)
    if getattr(m, 'bias', None) is not None: init.constant_(m.bias, 0.)
    for l in m.children(): init_cnn(l)

class ResBlock(nn.Module):
    def __init__(self, ni, no, expansion, s=1):
        super().__init__()
        ni *= expansion
        layers = [conv_layer(ni, no, 3, s),
                  conv_layer(no, no*expansion, 3, 1, zero_bn=True, act=False) 
        ] if expansion == 1 else [
            conv_layer(ni, no, 1, 1),
            conv_layer(no, no, 3, s),
            conv_layer(no, no*expansion, 1, 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        self.idconv = noop if ni == no*expansion else conv_layer(ni, no*expansion, 1, 1, act=False)  
        self.pool = noop if s == 1 else nn.AvgPool2d(2,ceil_mode=True)

    def forward(self, x):
        return act_func(self.convs(x) + self.idconv(self.pool(x)))

class XResNet(nn.Sequential):

    @classmethod
    def create(cls, expansion,  layers, c_in=3, c_out=10,resize=cifar_resize,compression_factor=1):
        nbs = [c_in, 32,64,64]
        stem = [conv_layer(nbs[i], nbs[i+1], 3, 2 if i==0 else 1,False) 
                for i in range(3)] 

        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        nbs = [64//expansion, 64, 128//compression_factor, 256//compression_factor, 512//compression_factor]
        res_layers = [cls._make_layer(nbs[i], nbs[i+1], expansion, 1 if i==0 else 2, l)
            for i,l in enumerate(layers) ]
        
        layers = [Lambda(resize), *stem, maxpool, *res_layers]
        layers.extend([nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(nbs[-1]*expansion, c_out) ])
        
        resnet = cls(*layers)        
        init_cnn(resnet)
        return resnet

    @staticmethod
    def _make_layer(ni, no, expansion, s, number_block):
        return nn.Sequential(*[ResBlock(ni if i==0 else no, no, expansion, s=s if i==0 else 1) 
            for i in range(number_block)] )


def xresnet18 (mask=1,**kwargs):
    global randommask 
    randommask = mask
    return XResNet.create(1, [2, 2,  2, 2], **kwargs)

def xresnet34 (**kwargs): return XResNet.create(1, [3, 4,  6, 3], **kwargs)
def xresnet50 (**kwargs): return XResNet.create(4, [3, 4,  6, 3], **kwargs)
def xresnet100 (**kwargs): return XResNet.create(4, [6, 8,  12, 6], **kwargs)
def xresnet101(**kwargs): return XResNet.create(4, [3, 4, 23, 3], **kwargs)
def xresnet152(**kwargs): return XResNet.create(4, [3, 8, 36, 3], **kwargs)


