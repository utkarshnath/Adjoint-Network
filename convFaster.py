from torch.nn.parameter import Parameter
import torch
from torch.autograd import Function
from torch import tensor, nn
import math
import torch.nn.functional as F
import time
from mask import *

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)

class conv2dFirstLayer(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,*kargs,**kwargs):
        super(conv2dFirstLayer, self).__init__(in_channels,out_channels,kernel_size,padding,stride,*kargs, **kwargs)
        self.padding = (padding,padding)
        self.stride = (stride,stride)

    def forward(self,input):
        a = F.conv2d(input,self.weight,self.bias,self.stride,self.padding,self.dilation, self.groups)
        concatinatedTensor = torch.cat([a, a], dim=0)
        return concatinatedTensor

class conv2dFaster(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,mask_layer,compression_factor=1,masking_factor=None,*kargs,**kwargs):
        super(conv2dFaster, self).__init__(in_channels,out_channels,kernel_size,padding,stride,*kargs, **kwargs)
        self.padding = (padding,padding)
        self.stride = (stride,stride)
        self.mask_layer = mask_layer
        self.out_channels = out_channels
        self.compression_factor = compression_factor
        if masking_factor!=None:
           self.mask = randomShape(kernel_size,kernel_size,masking_factor)
        else:
          self.mask = 1
        self.isFirst = True 

    def forward(self,input):
        l,_,_,_ = input.shape
        a = F.conv2d(input[:l//2],self.weight,self.bias,self.stride,self.padding)
        if self.mask_layer:
           b = F.conv2d(input[l//2:],self.weight*self.mask,self.bias,self.stride,self.padding)
           b[:,self.out_channels//self.compression_factor:] = 0
           concatinatedTensor = torch.cat([a, b], dim=0)
        else:
           concatinatedTensor = torch.cat([a, a], dim=0)

        return concatinatedTensor
        
class batchNorm(nn.Module):
    def __init__(self,num_features,*kargs,**kwargs):
        super(batchNorm,self).__init__(*kargs,**kwargs)
        self.num_features = num_features
        self.bn1 = nn.BatchNorm2d(num_features)
        self.bn2 = nn.BatchNorm2d(num_features)

    def forward(self,input):
        l,_,_,_ = input.shape
        a = self.bn1(input[:l//2])
        d = self.bn2(input[l//2:])
        concatinatedTensor = torch.cat([a, d], dim=0)
        return concatinatedTensor

class linear(nn.Linear):
    def __init__(self,in_features, out_features, parts=4, bias=True,*kargs,**kwargs):
        super(linear, self).__init__(in_features, out_features, bias=True,*kargs, **kwargs)

    def forward(self,input):
        l,_ = input.shape
        a = F.linear(input[:l//2], self.weight, self.bias)
        d = F.linear(input[l//2:], self.weight, self.bias)
        concatinatedTensor = torch.cat([a, d], dim=0)
        return concatinatedTensor

class MyCrossEntropy(nn.Module):
    def __init__(self,alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        l,_ = output.shape
        log_preds1 = F.log_softmax(output[:l//2], dim=-1)
        nll1 = F.nll_loss(log_preds1, target)
        
        prob1 = F.softmax(output[:l//2], dim=-1)
        prob2 = F.softmax(output[l//2:], dim=-1)
        kl = (prob1 * torch.log(1e-6 + prob1/(prob2+1e-6))).sum(1)
        
        return nll1 + self.alpha * kl.mean()


