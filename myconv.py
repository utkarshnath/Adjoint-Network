from torch.nn.parameter import Parameter
import torch
from torch.autograd import Function
from torch import tensor, nn
import math
import torch.nn.functional as F
import time


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
        out_channels,_,hf,wf = weight.shape
        output_size = (h-hf)//stride + 1
        
        unfolded_input = torch._C._nn.im2col(input, (hf,wf),(1,1),(0,0),(stride,stride))
        unfolded_weight = weight.view(out_channels,-1)
        out = unfolded_weight @ unfolded_input
        out = out.view(N,out_channels,output_size,output_size)
        out = out[:,:,:,:] + bias[None,:,None,None]
        end = time.time()
        #print("forward ",end-start)
        return out
     
    @staticmethod
    def backward(context,grad_output):
        start = time.time()
        input,weight,bias= context.saved_tensors
        stride = context.stride
        padding = context.padding
        mask = context.mask
        grad_input, grad_weight = torch.zeros(input.shape).cuda(),torch.zeros(weight.shape).cuda() #cuda for both
        out_channels,in_channels,k,_ = weight.shape
        n,f,h0,w0 = grad_output.shape
        # need padding of (kernel-1) size and each output should be seperated by (stride-1)
        length = 2*(k-1) + h0 + (h0-1)*(stride-1)
        out = torch.zeros(n,f,length,length).cuda()
        outLength = (length - k) + 1
        end = time.time()
        #print('Initial time ',end-start) 


        grad_bias = grad_output.sum((0,2,3))        
        end1 = time.time()
        #print('bias derivate ',end1-end)

        for i in range(0,k):
            for j in range(0,k):
                grad_weight[:,:,i,j] = ((input[:,None,:,i:i+h0*stride:stride,j:j+w0*stride:stride] * grad_output[:,:,None,:,:]).sum((0,3,4)))
        end2 = time.time()
        #print('weight derivate ',end2-end1)

        '''
        _,_,l,_ = input.shape
        t = (l-k)//stride
        unfolded_input = input.permute(1,0,2,3)
        unfolded_input = torch._C._nn.im2col(unfolded_input, (h0,w0),(stride,stride),(0,0),(1,1))
        unfolded_grad_output = grad_output.permute(1,0,2,3)
        unfolded_grad_output = unfolded_grad_output.reshape(f,-1)
        grad_weight = unfolded_grad_output @ unfolded_input
        grad_weight = grad_weight.view(weight.shape)
       
        
        _,_,l,_ = input.shape
        t = (l-k)//stride
        unfolded_input = torch._C._nn.im2col(input[:,:,0:k+t*stride,0:k+t*stride], (h0,w0),(stride,stride),(0,0),(1,1))
        unfolded_grad_output = grad_output.view(n,-1)
        print(unfolded_grad_output.shape)
        print(unfolded_input.shape)
        grad_weight = unfolded_grad_output @ unfolded_input
        print(grad_weight.shape)
        print(weight.shape)
        grad_weight = grad_weight.view(weight.shape)
        #unfold = torch.nn.Unfold(kernel_size=(h0,w0), padding=0, dilation=stride)        
        #unfolded_input = unfold(input[:,:,0:k+t*stride,0:k+t*stride])
        #x,y,z = unfolded_input.shape
        #unfolded_input = unfolded_input.view(x,input.shape[1],h0*w0,z)

        #unfold = torch.nn.Unfold(kernel_size=(h0,w0), padding=0, stride=1)
        #unfolded_grad_output = unfold(grad_output).transpose(2, 1)
        
        #a,b,c = unfolded_grad_output.shape
        #unfolded_grad_output = unfolded_grad_output.view(a,b,c//(h0*w0),h0*w0)
        #grad_weight = (unfolded_grad_output[:,:,:,None,:,None] * unfolded_input[:,None,None,:,:,:]).sum((0,1,4))
        #grad_weight = grad_weight.view(weight.shape)
        ''' 
         
        weight = weight*mask
        weight = torch.flip(weight,[2,3])
        '''
        weight = weight.view(out_channels,in_channels,-1)
        if(k==3):
           for i in range(4):
               temp = weight[:,:,i] 
               weight[:,:,i] = weight[:,:,8-i]
               weight[:,:,8-i] = temp
        '''
        #print("flip ",end4-end1)
        if h0==length:
           out = grad_output
        else:
           out[:,:,k-1:1-k:stride,k-1:1-k:stride] = grad_output

        end3 = time.time()
        #print("check point",end3-end2)
        '''
        #unfold = torch.nn.Unfold(kernel_size=(k,k), padding=0, stride=1)
        #unfolded_weight = unfold(weight).transpose(2, 1)
        unfolded_weight = torch._C._nn.im2col(weight, (k,k),(1,1),(0,0),(1,1)).transpose(2, 1)
        unfolded_weight = unfolded_weight.view(out_channels,1,in_channels,k*k)
        #unfolded_out = unfold(out)
        unfolded_out = torch._C._nn.im2col(out, (k,k),(1,1),(0,0),(1,1))
        a,b,c = unfolded_out.shape
        #unfolded_weight = weight.view(out_channels,-1)
        #print(unfolded_weight.shape)
        #print(unfolded_out.shape)
        #grad_input1 = unfolded_weight @ unfolded_out
        '''
        unfolded_out = torch._C._nn.im2col(out, (k,k),(1,1),(0,0),(1,1))
        weight = weight.permute(1,0,2,3)
        unfolded_weight = weight.reshape(in_channels,-1)
        grad_input1 = unfolded_weight @ unfolded_out
        #print('input grad ',end3-end2)
        #unfolded_out = unfolded_out.view(a,out_channels,k*k,c)
        #grad_input1 = (unfolded_weight[None,:,:,:,:,None] * unfolded_out[:,:,None,None,:,:]).sum((1,2,4))
        grad_input[:,:,0:outLength,0:outLength] =  grad_input1.view(n,in_channels,outLength,outLength)
        end4 = time.time()
        #print("input gradient ",end4-end3)        

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


