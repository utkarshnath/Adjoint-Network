from torch.nn.parameter import Parameter
import torch
from torch.autograd import Function
from torch import tensor, nn
import math
import torch.nn.functional as F
import time
from mask import *
import torch.distributions.gumbel as gumbel

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


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

class conv2dFirstLayer(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,*kargs,**kwargs):
        super(conv2dFirstLayer, self).__init__(in_channels,out_channels,kernel_size,padding,stride,*kargs, **kwargs)
        self.padding = (padding,padding)
        self.stride = (stride,stride)

    def forward(self,input, epoch=None, latency=0, prev_g_weight=0):
        a = F.conv2d(input,self.weight,self.bias,self.stride,self.padding,self.dilation, self.groups)
        concatinatedTensor = torch.cat([a, a], dim=0)
        return concatinatedTensor, latency, prev_g_weight

class conv2dAdjoint(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,mask_layer, architecture_search=False, compression_factor=1,masking_factor=None,*kargs,**kwargs):
        super(conv2dAdjoint, self).__init__(in_channels,out_channels,kernel_size,padding,stride,*kargs, **kwargs)
        self.gumbel_weight = Parameter(torch.rand(4))
        self.gumbel_noise = Parameter(sample_gumbel(self.gumbel_weight.size()))
        self.gumbel_noise.requires_grad = False
        self.padding = (padding,padding)
        self.stride = (stride,stride)
        self.mask_layer = mask_layer
        self.out_channels = out_channels
        self.compression_factor = compression_factor
        self.architecture_search = architecture_search
        if masking_factor!=None:
           self.mask = randomShape(kernel_size,kernel_size,masking_factor)
        else:
          self.mask = 1
        if not self.architecture_search:
            self.gumbel_weight.requires_grad = False

    def forward(self,input, epoch=None, latency=0, prev_g_weight=0):
        l,_,_,_ = input.shape
        if self.architecture_search:
            g_weight = gumbel_softmax(self.gumbel_weight, self.gumbel_noise, 15*((0.956)**epoch), False)
        else:
            g_weight = gumbel_softmax(self.gumbel_weight, self.gumbel_noise, 0.01, True)
        
        #g_weight = F.gumbel_softmax(self.gumbel_weight, 5*((0.956)**epoch), False)
        a = F.conv2d(input[:l//2],self.weight,self.bias,self.stride,self.padding)
        
        if self.mask_layer:
           b = F.conv2d(input[l//2:],self.weight*self.mask,self.bias,self.stride,self.padding)
            
           b1 = torch.clone(b)
           b1[:,self.out_channels//1:] = 0
           b1 = b1*g_weight[0]
           
           b2 = torch.clone(b)
           b2[:,self.out_channels//2:] = 0
           b2 = b2*g_weight[1]
 
           b3 = torch.clone(b)
           b3[:,self.out_channels//4:] = 0
           b3 = b3*g_weight[2]

           b4 = torch.clone(b)
           b4[:,self.out_channels//8:] = 0
           b4 = b4*g_weight[3]

           #b5 = torch.clone(b)
           #b5[:,self.out_channels//64:] = 0
           #b5 = b5*g_weight[4]

           if type(prev_g_weight)==int:
              c_in = self.weight.shape[1]
           else:
              c_in = ((self.weight.shape[1]//1)*prev_g_weight[0] + 
                     (self.weight.shape[1]//2)*prev_g_weight[1] + 
                     (self.weight.shape[1]//4)*prev_g_weight[2] + 
                     (self.weight.shape[1]//8)*prev_g_weight[3])
                     #(self.weight.shape[1]//64)*prev_g_weight[4]) 
           
           h = input.shape[2]
           w = input.shape[3]
           k = self.weight.shape[2]

           c_out = (((self.out_channels//1)**1)*g_weight[0] +
                    ((self.out_channels//2)**1)*g_weight[1] +
                    ((self.out_channels//4)**1)*g_weight[2] +
                    ((self.out_channels//8)**1)*g_weight[3])
                    #((self.out_channels//64)**1)*g_weight[4])

           latency += (k * k * h * w * c_in * c_out)

           concatinatedTensor = torch.cat([a, b1+b2+b3+b4], dim=0)
           return concatinatedTensor, latency, g_weight
        else:
           concatinatedTensor = torch.cat([a, a], dim=0)
           
        return concatinatedTensor, latency, 0
        
class batchNorm(nn.Module):
    def __init__(self,num_features,*kargs,**kwargs):
        super(batchNorm,self).__init__(*kargs,**kwargs)
        self.num_features = num_features
        self.bn1 = nn.BatchNorm2d(num_features)
        self.bn2 = nn.BatchNorm2d(num_features)

    def forward(self,input, epoch=None, latency=0, prev_g_weight=0):
        l,_,_,_ = input.shape
        a = self.bn1(input[:l//2])
        d = self.bn2(input[l//2:])
        concatinatedTensor = torch.cat([a, d], dim=0)
        return concatinatedTensor, latency, prev_g_weight

class linear(nn.Linear):
    def __init__(self,in_features, out_features, parts=4, bias=True,*kargs,**kwargs):
        super(linear, self).__init__(in_features, out_features, bias=True,*kargs, **kwargs)

    def forward(self,input):
        l,_ = input.shape
        a = F.linear(input[:l//2], self.weight, self.bias)
        d = F.linear(input[l//2:], self.weight, self.bias)
        concatinatedTensor = torch.cat([a, d], dim=0)
        return concatinatedTensor

class AdjointLoss(nn.Module):
    def __init__(self,alpha=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = 1e-17

    def forward(self, output, target, latency, architecture_search):
        l,_ = output.shape
        log_preds1 = F.log_softmax(output[:l//2], dim=-1)
        nll1 = F.nll_loss(log_preds1, target)
        
        prob1 = F.softmax(output[:l//2], dim=-1)
        prob2 = F.softmax(output[l//2:], dim=-1)
        kl = (prob1 * torch.log(1e-6 + prob1/(prob2+1e-6))).sum(1)
        # print(nll1, kl.mean(), self.gamma * latency)
        if architecture_search:
            return nll1 + self.alpha * (kl.mean() + self.gamma * latency)
        else:
            return nll1 + self.alpha * kl.mean()

class TeacherStudentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, teacher_output, student_output, target):
        log_preds = F.log_softmax(student_output, dim=-1)
        nll = F.nll_loss(log_preds, target)

        prob1 = F.softmax(teacher_output, dim=-1)
        prob2 = F.softmax(student_output, dim=-1)
        kl = (prob1 * torch.log(1e-6 + prob1/(prob2+1e-6))).sum(1)

        return nll +  kl.mean()
