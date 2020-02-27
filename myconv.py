from torch.nn.parameter import Parameter
import torch
from torch.autograd import Function
from torch import tensor, nn
import math
import torch.nn.functional as F

class convolutionFunction(Function):

    @staticmethod
    def forward(context,input,weight,bias,padding,stride,mask):
        context.padding = padding
        context.stride = stride
        context.mask = mask
        weight = weight*mask
        pad = (padding,padding,padding,padding)
        input = F.pad(input,pad,"constant",0).cuda()
        context.save_for_backward(input,weight,bias)
        N,C,h,w = input.shape
        out_channels,_,hf,wf = weight.shape
        output_size = (h-hf)//stride + 1

        unfold = torch.nn.Unfold(kernel_size=(hf,wf), padding=0, stride=stride)
        unfolded_input = unfold(input)
        unfold = torch.nn.Unfold(kernel_size=(hf,wf), padding=0, stride=1)
        unfolded_weight = unfold(weight).transpose(2, 1)
        out = (unfolded_weight[None,:,:,:,None]*unfolded_input[:,None,None,:,:]).sum((2,3))
        out = out.view(N,out_channels,output_size,output_size)
        out = out[:,:,:,:] + bias[None,:,None,None]
        return out
     
    @staticmethod
    def backward(context,grad_output):
        input,weight,bias= context.saved_tensors
        stride = context.stride
        padding = context.padding
        mask = context.mask

        grad_bias = grad_output.sum((0,2,3))
        grad_input, grad_weight = torch.zeros(input.shape).cuda(),torch.zeros(weight.shape).cuda() #cuda for both
        out_channels,in_channels,k,_ = weight.shape
        n,f,h0,w0 = grad_output.shape

        for i in range(0,k):
            for j in range(0,k):
                grad_weight[:,:,i,j] = ((input[:,None,:,i:i+h0*stride:stride,j:j+w0*stride:stride] * grad_output[:,:,None,:,:]).sum((0,3,4)))
        ''' 
        _,_,l,_ = input.shape
        t = (l-k)//stride
        unfold = torch.nn.Unfold(kernel_size=(h0,w0), padding=0, dilation=stride)        
        unfolded_input = unfold(input[:,:,0:k+t*stride,0:k+t*stride])
        x,y,z = unfolded_input.shape
        unfolded_input = unfolded_input.view(x,input.shape[1],h0*w0,z)

        unfold = torch.nn.Unfold(kernel_size=(h0,w0), padding=0, stride=1)
        unfolded_grad_output = unfold(grad_output).transpose(2, 1)
        
        a,b,c = unfolded_grad_output.shape
        unfolded_grad_output = unfolded_grad_output.view(a,b,c//(h0*w0),h0*w0)
        grad_weight = (unfolded_grad_output[:,:,:,None,:,None] * unfolded_input[:,None,None,:,:,:]).sum((0,1,4))
        grad_weight = grad_weight.view(weight.shape)
        '''


        #grad_weight*=mask  - not doing this as weights get masked automatically in forward pass
        weight = weight*mask
        weight = torch.flip(weight,[2,3])
        # need padding of (kernel-1) size and each output should be seperated by (stride-1)
        length = 2*(k-1) + h0 + (h0-1)*(stride-1)
        out = torch.zeros(n,f,length,length).cuda()
        out[:,:,k-1:1-k:stride,k-1:1-k:stride] = grad_output


        _,_,h,w = input.shape
        outLength = (length - k) + 1
        unfold = torch.nn.Unfold(kernel_size=(k,k), padding=0, stride=1)
        unfolded_weight = unfold(weight).transpose(2, 1)
        unfolded_weight = unfolded_weight.view(out_channels,1,in_channels,k*k)
        unfolded_out = unfold(out)
        a,b,c = unfolded_out.shape
        unfolded_out = unfolded_out.view(a,out_channels,k*k,c)
        grad_input1 = (unfolded_weight[None,:,:,:,:,None] * unfolded_out[:,:,None,None,:,:]).sum((1,2,4))
        grad_input[:,:,0:outLength,0:outLength] =  grad_input1.view(n,in_channels,outLength,outLength)

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
        self.padding = padding
        self.stride = stride
        self.mask = mask

    def forward(self,input):
        return convolutionFunction().apply(input, self.weight,self.bias,self.padding,self.stride,self.mask)


