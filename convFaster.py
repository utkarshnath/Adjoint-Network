from torch.nn.parameter import Parameter
import torch
from torch.autograd import Function
from torch import tensor, nn
import math
import torch.nn.functional as F
import time



class conv2dFirstLayer(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,mask_layer,mask=1,parts=4,*kargs,**kwargs):
        super(conv2dFirstLayer, self).__init__(in_channels,out_channels,kernel_size,padding,stride,mask,*kargs, **kwargs)
        self.mask = torch.ones(parts,out_channels,in_channels,kernel_size,kernel_size).cuda()
        for i in range(1,parts):
            start = out_channels - i*out_channels//parts
            self.mask[i,start:out_channels] = 0
        self.padding = (padding,padding)
        self.stride = (stride,stride)
        self.mask_layer = mask_layer
        #x = out_channels//64
        #start = out_channels//x
        #self.mask[3,start:] = 0

    def forward(self,input):
        # print(self.mask[0].sum(),self.mask[1].sum(),self.mask[2].sum(),self.mask[3].sum())
        a = F.conv2d(input,self.weight * self.mask[0],self.bias,self.stride,self.padding,self.dilation, self.groups)
        #b = F.conv2d(input,self.weight * self.mask[1],self.bias,self.stride,self.padding,self.dilation, self.groups)
        #c = F.conv2d(input,self.weight * self.mask[2],self.bias,self.stride,self.padding,self.dilation, self.groups)
        if self.mask_layer:
           d = F.conv2d(input,self.weight * self.mask[3],self.bias,self.stride,self.padding,self.dilation, self.groups)
        else:
           d = F.conv2d(input,self.weight * self.mask[0],self.bias,self.stride,self.padding,self.dilation, self.groups)
        concatinatedTensor = torch.cat([a, d], dim=0)
        return concatinatedTensor



class conv2dFaster(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,mask_layer,mask=1,parts=4,*kargs,**kwargs):
        super(conv2dFaster, self).__init__(in_channels,out_channels,kernel_size,padding,stride,mask,*kargs, **kwargs)
        self.mask = torch.ones(parts,out_channels,in_channels,kernel_size,kernel_size).cuda()
        for i in range(1,parts):
            start = out_channels - i*out_channels//parts
            self.mask[i,start:out_channels] = 0
        self.padding = (padding,padding)
        self.stride = (stride,stride)
        self.mask_layer = mask_layer
        #if mask_layer:
        #   x = out_channels//32
        #   start = out_channels//x
        #   self.mask[3,start:out_channels] = 0

    def forward(self,input):
        l,_,_,_ = input.shape
        a = F.conv2d(input[:l//2],self.weight * self.mask[0],self.bias,self.stride,self.padding)
        #b = F.conv2d(input[l//4:l//2],self.weight * self.mask[1],self.bias,self.stride,self.padding)
        #c = F.conv2d(input[l//2:3*l//4],self.weight * self.mask[2],self.bias,self.stride,self.padding)
        if self.mask_layer:
           d = F.conv2d(input[l//2:],self.weight * self.mask[3],self.bias,self.stride,self.padding) 
        else:
           d = F.conv2d(input[l//2:],self.weight * self.mask[0],self.bias,self.stride,self.padding)
        concatinatedTensor = torch.cat([a, d], dim=0)
        return concatinatedTensor

class linear(nn.Linear):
    def __init__(self,in_features, out_features, parts=4, bias=True,*kargs,**kwargs):
        super(linear, self).__init__(in_features, out_features, bias=True,*kargs, **kwargs)

    def forward(self,input):
        l,_ = input.shape
        a = F.linear(input[:l//2], self.weight, self.bias)
        #b = F.linear(input[l//4:l//2], self.weight, self.bias)
        #c = F.linear(input[l//2:3*l//4], self.weight, self.bias)
        d = F.linear(input[l//2:], self.weight, self.bias)
        concatinatedTensor = torch.cat([a, d], dim=0)
        return concatinatedTensor

class MyCrossEntropy(nn.Module):
    def __init__(self,alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        l,_ = output.shape
        #print(output.shape,target.shape)
        log_preds1 = F.log_softmax(output[:l//2], dim=-1)
        log_preds2 = F.log_softmax(output[l//2:], dim=-1)
        #log_preds3 = F.log_softmax(output[l//2:3*l//4], dim=-1)
        #log_preds4 = F.log_softmax(output[3*l//4:], dim=-1)
        nll1 = F.nll_loss(log_preds1, target)
        nll2 = F.nll_loss(log_preds2, target)
        #nll3 = F.nll_loss(log_preds3, target)
        #nll4 = F.nll_loss(log_preds4, target)
        #return nll1 + self.alpha * ((nll1-nll2)**2).mean()
        return nll1 + self.alpha * (abs(nll1-nll2))
        #return nll1 + self.alpha *((output[:l//2] - output[l//2:])**2).mean()

class MyCrossEntropyFaster(nn.Module):
    def __init__(self,alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, output,output_big, target):
        l,_ = output.shape
        #print(output.shape,output_big.shape,target.shape)
        log_preds1 = F.log_softmax(output, dim=-1)
        log_preds2 = F.log_softmax(output_big, dim=-1)
        nll1 = F.nll_loss(log_preds1, target)
        nll2 = F.nll_loss(log_preds2, target)
        #return nll1 + self.alpha *(abs(output - output_big)).mean()
        return nll1 + self.alpha * ((nll1-nll2)**2).mean()
        return nll1 + self.alpha * (abs(nll1-nll2))
