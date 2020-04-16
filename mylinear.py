from torch.nn.parameter import Parameter
import torch
from torch.autograd import Function
from torch import tensor, nn
import math
import torch.nn.functional as F
import time

class linear(nn.Linear):
    def __init__(self,in_features, out_features, parts=4, bias=True,*kargs,**kwargs):
        super(linear, self).__init__(in_features, out_features, bias=True,*kargs, **kwargs)
        #self.weight = Parameter(torch.Tensor(in_features,out_features).cuda())
        #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))
        #self.bias = Parameter(torch.Tensor(out_features).cuda())

    def forward(self,input):
        l,_ = input.shape
        a = F.linear(input[:l//4], self.weight, self.bias)
        b = F.linear(input[l//4:l//2], self.weight, self.bias)
        c = F.linear(input[l//2:3*l//4], self.weight, self.bias)
        d = F.linear(input[3*l//4:], self.weight, self.bias)
        concatinatedTensor = torch.cat([a, b, c, d], dim=0) 
        return concatinatedTensor


