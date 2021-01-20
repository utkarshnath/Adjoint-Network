import torch.nn as nn
import torch.nn.init as init
from myconv import myconv2d
from mask import *
from adjointNetworkNAS import *

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def mnist_resize(x): return x.view(-1, 1, 28, 28)
def cifar_resize(x): return x.view(-1, 3, 32, 32)
def imagenet_resize(x): return x.view(-1, 3, 128, 128)
def noop(x, t=0, latency=0, prev_g_weight=0): return x, latency, prev_g_weight
def noop1(x): return x 

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

act_func = nn.ReLU()
class Relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
  
    def forward(self,x, t=0, latency=0, prev_g_weight=0):
        if type(x)==tuple:
           print('yes')
           return self.relu(x[0]), latency
        return self.relu(x),latency, prev_g_weight

act_func = Relu()

first = True

def conv(ni, no, ks, s=1, bias=False, expansion=4, architecture_search=False, compression_factor=4, masking_factor=None):
    if expansion==1:
       # resnet 18
       if no>64:
          mask_layer = True
       else:
          mask_layer = False
    else:
       # resnet 50
       if ni>64 and no>64:
          mask_layer = True
       else:
          mask_layer = False
    
    global first
    if first:
        first = False
        return conv2dFirstLayer(ni, no, kernel_size=ks, stride=s, padding=ks//2, bias=bias)

    return conv2dAdjoint(ni, no, kernel_size=ks, stride=s, padding=ks//2, bias=bias,mask_layer=mask_layer, architecture_search=architecture_search, compression_factor=compression_factor,masking_factor=masking_factor)

def conv_layer(ni, no, ks, s, zero_bn=False, act=True, expansion=1, architecture_search=False, compression_factor=4, masking_factor=None):
    bn = batchNorm(no)
    nn.init.constant_(bn.bn1.weight, 0. if zero_bn else 1.)
    nn.init.constant_(bn.bn2.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, no, ks=ks, s=s, expansion=expansion, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor), bn]
    if act: layers += [act_func]
    return mySequential(*layers)

def conv_layer_stem(ni, no, ks, s, zero_bn=False, act=True, expansion=1, architecture_search=False, compression_factor=4, masking_factor=None):
    bn = batchNorm(no)
    nn.init.constant_(bn.bn1.weight, 0. if zero_bn else 1.)
    nn.init.constant_(bn.bn2.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, no, ks=ks, s=s, expansion=expansion, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor), bn]
    if act: layers += [act_func]
    return nn.Sequential(*layers)

def init_cnn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
       init.kaiming_normal_(m.weight, a=1.0)
    if getattr(m, 'bias', None) is not None: init.constant_(m.bias, 0.)
    for l in m.children(): init_cnn(l)


class ResBlock(nn.Module):
    def __init__(self, ni, no, expansion, architecture_search, compression_factor, masking_factor, s=1):
        super().__init__()
        #print('Res ni ',ni,'no ',no)
        
        ni *= expansion
        layers = [conv_layer(ni, no, 3, s, expansion=expansion, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor),
                  conv_layer(no, no*expansion, 3, 1, zero_bn=True, act=False, expansion=expansion, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor)
        ] if expansion == 1 else [
            conv_layer(ni, no, 1, 1, expansion=expansion, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor),
            conv_layer(no, no, 3, s, expansion=expansion, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor),
            conv_layer(no, no*expansion, 1, 1, zero_bn=True, act=False, expansion=expansion, architecture_search=architecture_search,  compression_factor=compression_factor, masking_factor=masking_factor)
        ]
        self.convs = mySequential(*layers)
        self.idconv = noop if ni == no*expansion else conv_layer(ni, no*expansion, 1, 1, act=False, expansion=expansion, architecture_search=architecture_search,  compression_factor=compression_factor, masking_factor=masking_factor)
        self.pool = noop1 if s == 1 else nn.AvgPool2d(2,ceil_mode=True)
        

    def forward(self, x, epoch, latency, prev_g_weight):
        conv = self.convs(x, epoch, latency, prev_g_weight)
        idconv = self.idconv(self.pool(x), epoch, latency, prev_g_weight)
        return act_func(conv[0]+idconv[0])[0], conv[1]+idconv[1], conv[2]


class mySequential(nn.Sequential):
    def forward(self, input, epoch, latency, prev_g_weight):
        for module in self._modules.values():
            input, latency, prev_g_weight = module(input, epoch, latency, prev_g_weight)
        return input, latency, prev_g_weight

def make_layer(ni, no, expansion, s, number_block, architecture_search, compression_factor, masking_factor):
        return mySequential(*[ResBlock(ni if i==0 else no, no, expansion,architecture_search, compression_factor, masking_factor,s=s if i==0 else 1)
            for i in range(number_block)] )


class XResNet(nn.Module):
    def __init__(self, expansion,  layers, c_in=3, c_out=10, resize=cifar_resize, architecture_search=False, compression_factor=4, masking_factor=None):
        super().__init__()
        nbs = [c_in, 32,64,64]
        stem = [conv_layer(nbs[i], nbs[i+1], 3, 2 if i==0 else 1,False,expansion=expansion)
                for i in range(3)]

        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        nbs = [64//expansion, 64, 128, 256, 512]

        res_layers = [make_layer(nbs[i], nbs[i+1], expansion, 1 if i==0 else 2, l, architecture_search, compression_factor,masking_factor)
            for i,l in enumerate(layers) ]

        self.resize = Lambda(resize)
        self.stem = mySequential(*stem)
        self.pool = maxpool
        self.initial_layers = nn.Sequential(*[Lambda(resize), *stem, maxpool])
        self.res_layers = mySequential(*res_layers)
        self.final_layers = nn.Sequential(*[nn.AdaptiveAvgPool2d(1), Flatten(), linear(nbs[-1]*expansion, c_out)])

        init_cnn(self.initial_layers)
        init_cnn(self.final_layers)
        init_cnn(self.res_layers)

    def forward(self, x, epoch):
        x = self.resize(x)
        x, latency, prev_g_weight = self.stem(x, epoch, 0, 0)
        x = self.pool(x)
        x, latency, prev_g_weight = self.res_layers(x, epoch, 0, 0)
        x = self.final_layers(x)
        return x, latency


def xresnet_fast18 (**kwargs):return XResNet(1, [2, 2,  2, 2], **kwargs)
def xresnet_fast34 (**kwargs): return XResNet(1, [3, 4,  6, 3], **kwargs)
def xresnet_fast50 (**kwargs): return XResNet(4, [3, 4,  6, 3], **kwargs)
def xresnet_fast50X2 (**kwargs): return XResNet(4, [6, 8,  12, 6], **kwargs)
def xresnet_fast101(**kwargs): return XResNet(4, [3, 4, 23, 3], **kwargs)
def resnet_fast152(**kwargs): return XResNet(4, [3, 8, 36, 3], **kwargs)








