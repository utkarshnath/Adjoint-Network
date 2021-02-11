import torch.nn as nn
import torch.nn.init as init
from myconv import myconv2d
from mask import *
import torch.nn.functional as F
#import flops_utils
from model import xresnet18,xresnet50,xresnet101

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
    # For Resnet 50
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
        global index, compression_list, first
        super().__init__()
        ni *= expansion
        if ni>64 and no>64:
           layers = [conv_layer(ni, no, 3, s),
                     conv_layer(no, no*expansion, 3, 1, zero_bn=True, act=False)
           ] if expansion == 1 else [
               conv_layer(ni, no//compression_list[index], 1, 1) if first else conv_layer(ni//compression_list[index-1], no//compression_list[index], 1, 1),
               conv_layer(no//compression_list[index], no//compression_list[index+1], 3, s),
               conv_layer(no//compression_list[index+1], no*expansion//compression_list[index+2], 1, 1, zero_bn=True, act=False)
           ]
        
           self.convs = nn.Sequential(*layers)
           if ni == no*expansion:
              self.idconv = noop
              index = index+3
           else:
              if first:
                 self.idconv = conv_layer(ni, no*expansion//compression_list[index+3], 1, 1, act=False)
              else:
                 self.idconv = conv_layer(ni//compression_list[index-1], no*expansion//compression_list[index+3], 1, 1, act=False)
              index = index+4

           first = False
        else:
           layers = [conv_layer(ni, no, 3, s),
                    conv_layer(no, no*expansion, 3, 1, zero_bn=True, act=False)
           ] if expansion == 1 else [
               conv_layer(ni, no, 1, 1),
               conv_layer(no, no, 3, s),
               conv_layer(no, no*expansion, 1, 1, zero_bn=True, act=False)
           ]
           self.convs = nn.Sequential(*layers)
           self.idconv = noop if ni == no*expansion else conv_layer(ni, no*expansion, 1, 1, act=False) 
        #if self.idconv!= noop:
        #   print('rer')
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
        global compression_list, index
        layers = [Lambda(resize), *stem, maxpool, *res_layers]
        layers.extend([nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(nbs[-1]*expansion//compression_list[index-1], c_out) ])

        resnet = cls(*layers)
        init_cnn(resnet)
        return resnet

    @staticmethod
    def _make_layer(ni, no, expansion, s, number_block):
        return nn.Sequential(*[ResBlock(ni if i==0 else no, no, expansion, s=s if i==0 else 1)
            for i in range(number_block)] )

def xresnet18_custom (mask=1,**kwargs):
    global randommask
    randommask = mask
    return XResNet.create(1, [2, 2,  2, 2], **kwargs)

def xresnet34_custom (**kwargs): return XResNet.create(1, [3, 4,  6, 3], **kwargs)
def xresnet50_custom (**kwargs): return XResNet.create(4, [3, 4,  6, 3], **kwargs)
def xresnet101_custom(**kwargs): return XResNet.create(4, [3, 4, 23, 3], **kwargs)
def xresnet152_custom(**kwargs): return XResNet.create(4, [3, 8, 36, 3], **kwargs)

def gumbel_softmax(logits, gumbel_noise, temperature, hard=False):
    y = logits + gumbel_noise
    y = F.softmax(y / temperature, dim=-1)
    if not hard:
       return y
    else:
      idx = torch.argmax(y)
      y_hard = torch.zeros_like(y).cuda()
      y_hard.scatter_(0, idx, 1)
      return y_hard

def printModel(path):
    state_dict = torch.load(path)
    gumbel = None
    noise = None
    mask = False
    i = 0
    compression_list = []
    for k,v in state_dict.items():
        #if(len(v.shape)==4):
        #  print(v.shape)
        if(k.find('initial_layers')!=-1):
           continue
        if(len(v.shape)==4 and v.shape[0]>64 and v.shape[1]>64):
          mask = True
        if(k.find("gumbel_weight")!=-1):
          gumbel = v
        if(k.find("gumbel_noise")!=-1):
          index = torch.argmax(gumbel_softmax(gumbel, v, 0.01, True))
          # print(index, 2**(2+index))
          if mask:
             compression_list += [int(2**(1+index))]
    return compression_list
    
if __name__ == "__main__":
   compression_list = printModel('/scratch/un270/model/Adjoint-Experiments/Nas/updated_config/search_imagewoof_124_e19_X2/79.pt')
   print(compression_list)
   #print(len(compression_list))
   index = 0
   #model = xresnet50(c_out=100,compression_factor=1)
   #print(index)
   #model= xresnet_fast50(c_out=100, compression_factor=1, masking_factor=None, architectur_search=False)
   '''
   model = xresnet50(c_out=100,compression_factor=64)
   fest = flops_utils.FlopsEstimation(model)
   model.cuda()
   input = torch.zeros(1,3,224,224).cuda()
   with fest.enable():
        model(input)
        nparams, nflops = fest.get_flops()
        print(nparams, nflops)
    '''
