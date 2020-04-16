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
        parts = weight.shape[0]

        #weight = weight*mask
        N,C,h,w = input.shape
        parts,out_channels,_,k,k = weight.shape
        output_size = (h-k+2*padding)//stride + 1

        unfolded_input = torch._C._nn.im2col(input, (k,k),(1,1),(padding,padding),(stride,stride))
        unfolded_weight = weight.view(parts*out_channels,-1)
        out = unfolded_weight @ unfolded_input
        out = out.view(N,parts,out_channels,output_size,output_size)
        out = out.permute(1,0,2,3,4)
        out = out[:,:,:,:,:] + bias[:,None,:,None,None]
        context.save_for_backward(input,weight,bias,unfolded_input)
        return out

    @staticmethod
    def backward(context,grad_output):
        input,weight,bias,unfolded_input = context.saved_tensors
        #print("backward conv2dFirstLayer ")
        stride = context.stride
        padding = context.padding
        mask = context.mask
        n,_,h,w = input.shape
        parts,f,_,k,_ = weight.shape

        grad_bias = grad_output.sum((1,3,4))

        X_col = unfolded_input.permute(1,0,2)
        X_col = X_col.reshape(X_col.shape[0],-1)
        dout_reshaped = grad_output.permute(0,2,1,3,4).reshape(parts,f, -1)
        dW = dout_reshaped @ X_col.T
        grad_weight = dW.view(weight.shape) 

        # As this will only be used in first layer, no need of grad_input

        return input,grad_weight,grad_bias,None,None,None

class conv2dFirstLayer(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,mask=1,parts=4,*kargs,**kwargs):
        super(conv2dFirstLayer, self).__init__(in_channels,out_channels,kernel_size,padding,stride,mask,*kargs, **kwargs)
        #self.in_channels = in_channels
        #self.out_channels = out_channels
        #self.kernel_size = kernel_size
        #weight = torch.Tensor(out_channels,in_channels,kernel_size,kernel_size).cuda() #cuda
        #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))
        self.mask = torch.ones(parts,out_channels,in_channels,kernel_size,kernel_size).cuda()
        for i in range(1,parts):
            start = out_channels - i*out_channels//parts
            self.mask[i,start:out_channels] = 0
        #self.weight = Parameter(weight * self.mask)
        #self.bias = Parameter(torch.zeros(parts,out_channels).cuda())
        self.padding = (padding,padding)
        self.stride = (stride,stride)

    def forward(self,input):
        a = F.conv2d(input,self.weight * self.mask[0],self.bias,self.stride,self.padding,self.dilation, self.groups)
        b = F.conv2d(input,self.weight * self.mask[1],self.bias,self.stride,self.padding,self.dilation, self.groups)
        c = F.conv2d(input,self.weight * self.mask[2],self.bias,self.stride,self.padding,self.dilation, self.groups)
        d = F.conv2d(input,self.weight * self.mask[3],self.bias,self.stride,self.padding,self.dilation, self.groups)
        concatinatedTensor = torch.cat([a, b, c, d], dim=0) 
        return concatinatedTensor
