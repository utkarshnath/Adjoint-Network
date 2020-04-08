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
        N,C,h,w = input.shape
        out_channels,_,hf,wf = weight.shape
        output_size = (h-hf+2*padding)//stride + 1
        
        unfolded_input = torch._C._nn.im2col(input, (hf,wf),(1,1),(padding,padding),(stride,stride))
        unfolded_weight = weight.view(out_channels,-1)
        out = unfolded_weight @ unfolded_input
        out = out.view(N,out_channels,output_size,output_size)
        out = out[:,:,:,:] + bias[None,:,None,None]
        context.save_for_backward(input,weight,bias,unfolded_input)
        return out
     
    @staticmethod
    def backward(context,grad_output):
        input,weight,bias,unfolded_input = context.saved_tensors
        stride = context.stride
        padding = context.padding
        mask = context.mask
        n,_,h,w = input.shape
        f,_,k,_ = weight.shape

        grad_bias = grad_output.sum((0,2,3))        
        
        X_col = unfolded_input.permute(1,0,2)
        X_col = X_col.reshape(X_col.shape[0],-1)
        dout_reshaped = grad_output.permute(1, 0, 2, 3).reshape(f, -1)
        dW = dout_reshaped @ X_col.T
        grad_weight = dW.view(weight.shape)
        
 
        weight = weight*mask 
        weight = weight.reshape(f,-1)
        unfolded_grad_output = grad_output.permute(1,0,2,3)
        unfolded_grad_output = unfolded_grad_output.reshape(f,-1)
        dx = (weight.T)@unfolded_grad_output
        dx = dx.T
        dx = dx.reshape(n,-1,dx.shape[1])
        dx = dx.permute(0,2,1)
        grad_input = torch._C._nn.col2im(dx,(h,w),(k,k),(1,1),(padding,padding),(stride,stride))
        
        return grad_input,grad_weight,grad_bias,None,None,None

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


