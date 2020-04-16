from torch.nn.parameter import Parameter
import torch
from torch.autograd import Function
from torch import tensor, nn
import math
import torch.nn.functional as F
import time



class conv2dFirstLayer(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,mask=1,parts=4,*kargs,**kwargs):
        super(conv2dFirstLayer, self).__init__(in_channels,out_channels,kernel_size,padding,stride,mask,*kargs, **kwargs)
        self.mask = torch.ones(parts,out_channels,in_channels,kernel_size,kernel_size).cuda()
        for i in range(1,parts):
            start = out_channels - i*out_channels//parts
            self.mask[i,start:out_channels] = 0
        self.padding = (padding,padding)
        self.stride = (stride,stride)

    def forward(self,input):
        # print(self.mask[0].sum(),self.mask[1].sum(),self.mask[2].sum(),self.mask[3].sum())
        a = F.conv2d(input,self.weight * self.mask[0],self.bias,self.stride,self.padding,self.dilation, self.groups)
        b = F.conv2d(input,self.weight * self.mask[1],self.bias,self.stride,self.padding,self.dilation, self.groups)
        c = F.conv2d(input,self.weight * self.mask[2],self.bias,self.stride,self.padding,self.dilation, self.groups)
        d = F.conv2d(input,self.weight * self.mask[3],self.bias,self.stride,self.padding,self.dilation, self.groups)
        concatinatedTensor = torch.cat([a, b, c, d], dim=0)
        return concatinatedTensor



class conv2dFaster(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,mask=1,parts=4,*kargs,**kwargs):
        super(conv2dFaster, self).__init__(in_channels,out_channels,kernel_size,padding,stride,mask,*kargs, **kwargs)
        self.mask = torch.ones(parts,out_channels,in_channels,kernel_size,kernel_size).cuda()
        for i in range(1,parts):
            start = out_channels - i*out_channels//parts
            self.mask[i,start:out_channels] = 0
        self.padding = (padding,padding)
        self.stride = (stride,stride)

    def forward(self,input):
        l,_,_,_ = input.shape
        a = F.conv2d(input[:l//4],self.weight * self.mask[0],self.bias,self.stride,self.padding)
        b = F.conv2d(input[l//4:l//2],self.weight * self.mask[1],self.bias,self.stride,self.padding)
        c = F.conv2d(input[l//2:3*l//4],self.weight * self.mask[2],self.bias,self.stride,self.padding)
        d = F.conv2d(input[3*l//4:],self.weight * self.mask[3],self.bias,self.stride,self.padding) 
        concatinatedTensor = torch.cat([a, b, c, d], dim=0)
        return concatinatedTensor

class linear(nn.Linear):
    def __init__(self,in_features, out_features, parts=4, bias=True,*kargs,**kwargs):
        super(linear, self).__init__(in_features, out_features, bias=True,*kargs, **kwargs)

    def forward(self,input):
        l,_ = input.shape
        a = F.linear(input[:l//4], self.weight, self.bias)
        b = F.linear(input[l//4:l//2], self.weight, self.bias)
        c = F.linear(input[l//2:3*l//4], self.weight, self.bias)
        d = F.linear(input[3*l//4:], self.weight, self.bias)
        concatinatedTensor = torch.cat([a, b, c, d], dim=0)
        return concatinatedTensor

class MyCrossEntropy(nn.Module):
    def __init__(self,alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        l,_ = output.shape
        log_preds1 = F.log_softmax(output[:l//4], dim=-1)
        log_preds2 = F.log_softmax(output[l//4:l//2], dim=-1)
        log_preds3 = F.log_softmax(output[l//2:3*l//4], dim=-1)
        log_preds4 = F.log_softmax(output[3*l//4:], dim=-1)
        nll1 = F.nll_loss(log_preds1, target)
        nll2 = F.nll_loss(log_preds2, target)
        nll3 = F.nll_loss(log_preds3, target)
        nll4 = F.nll_loss(log_preds4, target)
        #return nll1 + torch.dist(output[:l//4],output[l//4:l//2]) + torch.dist(output[:l//4],output[l//2:3*l//4]) + torch.dist(output[:l//4],output[3*l//4:])
        return nll1 + self.alpha * (abs(nll1-nll2) + abs(nll1-nll3) + abs(nll1-nll4))
