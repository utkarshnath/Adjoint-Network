from helper import get_data_bunch
from run import Runner, Learn
from IPython.core.debugger import set_trace
import torch.nn.functional as F
from torch import tensor, nn, optim
from callback import AvgStatsCallback,CudaCallback
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
    def forward(context,input,weight,bias):
        context.save_for_backward(input,weight,bias)
        N,C,h,w = input.shape
        out_channels,_,hf,wf = weight.shape
        out = torch.Tensor(N,out_channels,h-hf+1,w-wf+1).cuda() # cuda
        outTest = torch.zeros(N,out_channels,h-hf+1,w-wf+1).cuda()
        for i in range(0,h-hf+1):
            for j in range(0,w-wf+1):
                out[:,:,i,j] = (input[:,None,:,i:i+hf,j:j+wf] * weight[None,:,:,:,:]).sum((2,3,4))
        out = out[:,:,:,:] + bias[None,:,None,None]
        #outTest = torch.zeros(512,8,24,24).cuda()
        return out
	
    @staticmethod
    def backward(context,grad_output):
        #print('backward called',grad_output.shape)
        input,weight,bias = context.saved_tensors
        grad_bias = grad_output.sum((0,2,3)) / batch_size
        grad_input, grad_weight = torch.Tensor(input.shape).cuda(),torch.Tensor(weight.shape).cuda() #cuda for both
        out_channels,in_channels,k,_ = weight.shape
        _,_,hf,wf = grad_output.shape
        for i in range(0,k):
            for j in range(0,k):
                grad_weight[:,:,i,j] = (input[:,None,:,i:i+hf,j:j+wf] * grad_output[:,:,None,:,:]).sum((0,3,4))
        
        grad_weight/=batch_size

        pad = (k-1,k-1,k-1,k-1)
        out = F.pad(grad_output, pad, "constant", 0).cuda() #cuda
        torch.flip(weight,[2,3])
        _,_,h0,w0 = out.shape
        for i in range(0,h0-k+1):
            for j in range(0,w0-k+1):
                grad_input[:,:,i,j] = (out[:,:,None,i:i+k,j:j+k] * weight[None,:,:,:,:]).sum((1,3,4))
        
        grad_input/=out_channels
        return grad_input,grad_weight,grad_bias

class myconv2d(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,*kargs,**kwargs):
        super(myconv2d, self).__init__(in_channels,out_channels,kernel_size,*kargs, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size).cuda()) #cuda
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))        

    def forward(self,input):
        #testInput = torch.randn(512,1,8,5,dtype=torch.double,requires_grad=True)
        #test = gradcheck(convolutionFunction.apply, (testInput,self.weight,self.bias), eps=1e-6, atol=1e-4)
        #print(test)
        return convolutionFunction().apply(input, self.weight,self.bias)
    

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
        myconv2d( 1, 8, 5, padding=2,stride=2), nn.ReLU(), #14
        myconv2d( 8,16, 3, padding=1,stride=2), nn.ReLU(), # 7
        myconv2d(16,32, 3, padding=1,stride=2), nn.ReLU(), # 4
        myconv2d(32,32, 3, padding=1,stride=2), nn.ReLU(), # 2
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(32,c)
    )

device = torch.device('cuda',0)
torch.cuda.set_device(device)

model = get_cnn_model(data)

opt = optim.SGD(model.parameters(), lr=0.2)
learn = Learn(model, opt, loss_func, data)
cbfs = [CudaCallback(),AvgStatsCallback()] #cuda
run = Runner(learn,cbs = cbfs)

'''# torch.cuda.tensor(512,1,8,5,requires_grad=True).normal_()
testInput = torch.randn(512,1,8,5,dtype=torch.double,requires_grad=True).cuda()
weight_test = Parameter(torch.randn(8,1,5,5,dtype=torch.double,requires_grad=True).cuda()) #cuda
nn.init.kaiming_uniform_(weight_test, a=math.sqrt(3))
bias_test = torch.randn(8,dtype=torch.double,requires_grad=True).cuda()
test = gradcheck(convolutionFunction.apply, (bias_test), eps=1e-4, atol=1e-4)
print(test)'''
run.fit(25)
