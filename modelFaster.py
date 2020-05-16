import torch.nn as nn
import torch.nn.init as init
from myconv import myconv2d
from mask import *
from convFaster import *

randommask = 1
device = None

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
    # For resnet 18
    #if no>64:
    #   mask_layer = True
    #else:
    #   mask_layer = False

    # For resnet 50
    if ni>64 and no>64:
       mask_layer = True
    else:
       mask_layer = False

    global first
    if first:
        first = False
        return conv2dFirstLayer(ni, no, kernel_size=ks, stride=s, padding=ks//2, bias=bias,mask_layer=mask_layer)

    return conv2dFaster(ni, no, kernel_size=ks, stride=s, padding=ks//2, bias=bias,mask_layer=mask_layer)

'''
def init_cnn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)): init.kaiming_normal_(m.weight, a=1.0)
    if getattr(m, 'bias', None) is not None: init.constant_(m.bias, 0.)
    for l in m.children(): init_cnn(l)
'''

def conv_layer(ni, no, ks, s, zero_bn=False, act=True,mask_layer=True):
    #print('conv ','ni ',ni,'no ',no,ks,s)
    bn = batchNorm(no)
    #bn = nn.BatchNorm2d(no)
    nn.init.constant_(bn.bn1.weight, 0. if zero_bn else 1.)
    nn.init.constant_(bn.bn2.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, no, ks=ks, s=s,mask_layer=mask_layer), bn]
    if act: layers += [act_func]
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, ni, no, expansion, s=1):
        super().__init__()
        #print('Res ni ',ni,'no ',no)
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
        #print(self.convs(x).shape)
        #print(self.idconv(self.pool(x)).shape)
        return act_func(self.convs(x) + self.idconv(self.pool(x)))

def accuracy_faster(out, yb):
    l,_ = out.shape
    return (torch.argmax(out[:l//2], dim=1)==yb).float().mean()
def accuracy1_faster(out, yb):
    l,_ = out.shape
    return (torch.argmax(out[l//2:], dim=1)==yb).float().mean()

def top_k_accuracy_faster(out, yb, k=5):
    l,_ = out.shape
    idx = out[:l//2].topk(k=k, dim=1)[1]
    yb = yb.unsqueeze(dim=1).expand_as(idx)
    return (yb == idx).max(dim=1)[0].float().mean()

def top_k_accuracy1_faster(out, yb, k=5):
    l,_ = out.shape
    idx = out[l//2:].topk(k=k, dim=1)[1]
    yb = yb.unsqueeze(dim=1).expand_as(idx)
    return (yb == idx).max(dim=1)[0].float().mean()

def nll(out, yb):
    l,_ = out.shape
    log_preds = F.log_softmax(out[:l//2], dim=-1)
    nll1 = F.nll_loss(log_preds, yb)
    return nll1

class mySequential(nn.Sequential):
    def forward(self, input,target):
        for module in self._modules.values():
            input = module(input)

        output = input
        l,_ = output.shape
        log_preds1 = F.log_softmax(output[:l//2], dim=-1)
        log_preds2 = F.log_softmax(output[l//2:], dim=-1)

        nll1 = F.nll_loss(log_preds1, target)

        prob1 = F.softmax(output[:l//2], dim=-1)
        prob2 = F.softmax(output[l//2:], dim=-1)
        kl = (prob1 * torch.log(1e-6+prob1/(prob2+1e-6))).sum(1)

        global device
        loss =  torch.tensor([nll1 + kl.mean()]).to(device)
        print(loss)
        top1_big = accuracy_faster(output,target)
        top5_big = top_k_accuracy_faster(output,target)
        top1_small = accuracy1_faster(output,target)
        top5_small = top_k_accuracy1_faster(output,target)
        
        return loss,[nll(output,target),top1_big,top5_big,top1_small,top5_small]

class XResNet(nn.Sequential):

    @classmethod
    def create(cls, expansion,  layers, c_in=3, c_out=10, resize=cifar_resize):
        nbs = [c_in, 16,64,64]
        stem = [conv_layer(nbs[i], nbs[i+1], 3, 2 if i==0 else 1,False)
                for i in range(2)]

        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        nbs = [64//expansion, 64, 128, 256, 512]
        res_layers = [cls._make_layer(nbs[i], nbs[i+1], expansion, 1 if i==0 else 2, l)
            for i,l in enumerate(layers) ]

        layers = [Lambda(resize), *stem, maxpool, *res_layers]
        layers.extend([nn.AdaptiveAvgPool2d(1), Flatten(), linear(nbs[-1]*expansion, c_out) ])

        resnet = cls(*layers)
        #init_cnn(resnet)        
        return resnet

    @staticmethod
    def _make_layer(ni, no, expansion, s, number_block):
        return nn.Sequential(*[ResBlock(ni if i==0 else no, no, expansion, s=s if i==0 else 1)
            for i in range(number_block)] )


def xresnet_fast18 (mask=1,**kwargs):
    global randommask
    randommask = mask
    return XResNet.create(1, [2, 2,  2, 2], **kwargs)

def xresnet_fast34 (**kwargs): return XResNet.create(1, [3, 4,  6, 3], **kwargs)
def xresnet_fast50 (device1,**kwargs):
    global device
    device = device1
    return XResNet.create(4, [3, 4,  6, 3], **kwargs)
def xresnet_fast101(**kwargs): return XResNet.create(4, [3, 4, 23, 3], **kwargs)
def resnet_fast152(**kwargs): return XResNet.create(4, [3, 8, 36, 3], **kwargs)
