from torch.nn.parameter import Parameter
import torch
from torch.autograd import Function
from torch import tensor, nn
import math
import torch.nn.functional as F
import time

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)

class convolutionFunction(Function):

    @staticmethod
    def forward(context,input,weight,bias,padding,stride,mask):
        start = time.time()
        context.padding = padding
        context.stride = stride
        context.mask = mask
        weight = weight*mask
        pad = (padding,padding,padding,padding)
        input = F.pad(input,pad,"constant",0).cuda()
        context.save_for_backward(input,weight,bias)
        N,C,h,w = input.shape
        print(input.shape)
        print(weight.shape)
        out_channels,_,hf,wf = weight.shape
        output_size = (h-hf)//stride + 1
        unfolded_input = torch._C._nn.im2col(input, (hf,wf),(1,1),(0,0),(stride,stride))
        unfolded_weight = weight.view(out_channels,-1)
        out = unfolded_weight @ unfolded_input
        out = out.view(N,out_channels,output_size,output_size)
        out = out[:,:,:,:] + bias[None,:,None,None]
        end = time.time()
        return out
     
    @staticmethod
    def backward(context,grad_output):
        start = time.time()
        input,weight,bias= context.saved_tensors
        stride = context.stride
        padding = context.padding
        mask = context.mask
        _,_,h,w = input.shape
        out_channels,in_channels,k,_ = weight.shape
        n,f,h0,w0 = grad_output.shape
        # need padding of (kernel-1) size and each output should be seperated by (stride-1)
        # length = 2*(k-1) + h0 + (h0-1)*(stride-1)
        # out = torch.zeros(n,f,length,length).cuda()
        # outLength = (length - k) + 1
        end = time.time()
        # print('Initial time ',end-start) 

        grad_bias = grad_output.sum((0,2,3))        
        end1 = time.time()
        # print('bias derivate ',end1-end)

        
        unfolded_input = input.permute(1,0,2,3)
        unfolded_input = torch._C._nn.im2col(unfolded_input, (h0,w0),(1,1),(0,0),(stride,stride))
        unfolded_grad_output = grad_output.permute(1,0,2,3)
        unfolded_grad_output = unfolded_grad_output.reshape(f,-1)
        grad_weight = unfolded_grad_output @ unfolded_input
        grad_weight = grad_weight.permute(1,0,2)
        grad_weight = grad_weight.view(weight.shape)
         
        weight = weight*mask 
        weight = weight.reshape(f,-1)
        unfolded_grad_output = grad_output.permute(1,0,2,3)
        unfolded_grad_output = unfolded_grad_output.reshape(f,-1)
        dx = (weight.T)@unfolded_grad_output
        dx = dx.T
        dx = dx.reshape(n,-1,dx.shape[1])
        dx = dx.permute(0,2,1)
        grad_input = torch._C._nn.col2im(dx,(h,w),(k,k),(1,1),(0,0),(stride,stride))
        end3 = time.time()
        if padding==0:
           return grad_input,grad_weight,grad_bias,None,None,None
        return grad_input[:,:,padding:-padding,padding:-padding],grad_weight,grad_bias,None,None,None

class myconv2d(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,mask=1,*kargs,**kwargs):
        super(myconv2d, self).__init__(in_channels,out_channels,kernel_size,padding,stride,mask,*kargs, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size).cuda()) #cuda
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))
        self.bias = Parameter(torch.zeros(out_channels).cuda())
        self.padding = padding
        self.stride = stride
        self.mask = mask

    def forward(self,input):
        return convolutionFunction().apply(input, self.weight,self.bias,self.padding,self.stride,self.mask)


