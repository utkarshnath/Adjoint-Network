from helper import get_data_bunch
from run import Runner, Learn
from IPython.core.debugger import set_trace
import torch.nn.functional as F
from torch import tensor, nn, optim
from callback import AvgStatsCallback,CudaCallback,GradientPrintCallback
from torch.autograd import Function
import torch
from torch.nn.parameter import Parameter 
# from .. import init
import math
from torch.autograd import gradcheck

batch_size = 512
c = 10
data = get_data_bunch(batch_size)
loss_func = F.cross_entropy


class convolutionFunction(Function):

    @staticmethod
    def forward(context,input,weight,bias,padding,stride,mask):
        context.padding = padding
        context.stride = stride
        context.mask = mask
        weight = weight*mask
        pad = (padding,padding,padding,padding)
        input = F.pad(input,pad,"constant",0).cuda()
        #print('Input shape',input.shape)
        context.save_for_backward(input,weight,bias)
        N,C,h,w = input.shape
        out_channels,_,hf,wf = weight.shape
        output_size = (h-hf)//stride + 1
        out = torch.Tensor(N,out_channels,output_size,output_size).cuda() # cuda
        for i in range(0,output_size):
            for j in range(0,output_size):
                istart = i*stride
                jstart = j*stride
                out[:,:,i,j] = (input[:,None,:,istart:istart+hf,jstart:jstart+wf] * weight[None,:,:,:,:]).sum((2,3,4))
        out = out[:,:,:,:] + bias[None,:,None,None]
        #print("weight shape",weight.shape)
        #print("output shpae",out.shape)
        return out
	
    @staticmethod
    def backward(context,grad_output):
        #print('output',grad_output.shape)
        input,weight,bias= context.saved_tensors
        stride = context.stride
        padding = context.padding
        mask = context.mask
        #print("Backward method")
        grad_bias = grad_output.sum((0,2,3)) 
        grad_input, grad_weight = torch.Tensor(input.shape).cuda(),torch.Tensor(weight.shape).cuda() #cuda for both
        out_channels,in_channels,k,_ = weight.shape
        _,_,hf,wf = grad_output.shape
        for i in range(0,k):
            for j in range(0,k):
                grad_weight[:,:,i,j] = ((input[:,None,:,i:i+hf*stride:stride,j:j+wf*stride:stride] * grad_output[:,:,None,:,:]).sum((0,3,4)))
        #grad_weight*=mask
        pad = (k-1,k-1,k-1,k-1)
        out = F.pad(grad_output, pad, "constant", 0).cuda() #cuda
        weight = torch.flip(weight,[2,3])
        _,_,h0,w0 = out.shape
        for i in range(0,h0-k+1):
            for j in range(0,w0-k+1):
                grad_input[:,:,i,j] = (out[:,:,None,i:i+k,j:j+k] * weight[None,:,:,:,:]).sum((1,3,4)) 
        #print('input grad',grad_input.shape)
        #print('weight grad',grad_weight.shape)
        #print('bias grad',grad_bias.shape)
        if padding==0:
           return grad_input,grad_weight,grad_bias,None,None,None
        return grad_input[:,:,padding:-padding,padding:-padding],grad_weight,grad_bias,None,None,None

class myconv2d(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,*kargs,**kwargs):
        super(myconv2d, self).__init__(in_channels,out_channels,kernel_size,padding,stride,*kargs, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size).cuda()) #cuda
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))
        self.padding = padding
        self.stride = stride
        mask5,mask3 = torch.ones(5,5).cuda(),torch.ones(3,3).cuda()
        mask5[0][0] = mask5[0][1] = mask5[0][3] = mask5[0][4] = mask5[4][0] = mask5[4][1] = mask5[4][3] = mask5[4][4] = mask5[1][0] = mask5[1][4] = mask5[3][0] = mask5[3][4] = 0
        mask3[0][0] = mask3[0][2] = mask3[2][0] = mask3[2][2] = 0
        if(kernel_size==3):
           self.mask = mask3
        else:
           self.mask = mask5

    def forward(self,input):
        return convolutionFunction().apply(input, self.weight,self.bias,self.padding,self.stride,self.mask)
    

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def flatten(x): return x.view(x.shape[0], -1)

def mnist_resize(x): return x.view(-1, 1, 28, 28)

def get_cnn_model(data):
    return nn.Sequential(
        Lambda(mnist_resize),
        myconv2d( 1, 8, 5, padding=0,stride=1), nn.ReLU(), #14
        myconv2d( 8,16, 3, padding=0,stride=1), nn.ReLU(), # 7
        myconv2d(16,32, 3, padding=0,stride=1), nn.ReLU(), # 4
        myconv2d(32,32, 3, padding=0,stride=1), nn.ReLU(), # 2
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(32,c)
    )

def printBackward(module,input,output):
    print("inputgrad",input)
    print("outputgrad",output)

device = torch.device('cuda',0)
torch.cuda.set_device(device)

model = get_cnn_model(data)

opt = optim.SGD(model.parameters(), lr=0.1)
learn = Learn(model, opt, loss_func, data)
cbfs = [CudaCallback(),AvgStatsCallback(),GradientPrintCallback()] #cuda
run = Runner(learn,cbs = cbfs)
#print(test)
run.fit(35)
