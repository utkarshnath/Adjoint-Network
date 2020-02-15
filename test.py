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

batch_size = 2
c = 8
data = get_data_bunch(batch_size)
loss_func = F.cross_entropy

firstlayer = True
weight = torch.Tensor(8,1,5,5).cuda()
bias = torch.Tensor(8).cuda()

class convolutionFunction(Function):

    @staticmethod
    def forward(context,input,weight,bias=None):
        context.save_for_backward(input,weight,bias)
        N,C,h,w = input.shape
        out_channels,_,hf,wf = weight.shape
        out = torch.Tensor(N,out_channels,h-hf+1,w-wf+1).cuda() # cuda
        for i in range(0,h-hf+1):
            for j in range(0,w-wf+1):
                out[:,:,i,j] = (input[:,None,:,i:i+hf,j:j+wf] * weight[None,:,:,:,:]).sum((2,3,4))
        out = out[:,:,:,:] + bias[None,:,None,None]
        return out

    @staticmethod
    def backward(context,grad_output):
        input,weight,bias = context.saved_tensors
        grad_bias = grad_output.sum((0,2,3)) / batch_size
        
        grad_input, grad_weight = torch.Tensor(input.shape).cuda(),torch.Tensor(weight.shape).cuda() #cuda for both
        out_channels,in_channels,k,_ = weight.shape
        _,_,hf,wf = grad_output.shape
        for i in range(0,k):
            for j in range(0,k):
                # print(input[:,c,i:i+hf,j:j+wf].shape, grad_output[:,f,:,:].shape)
                grad_weight[:,:,i,j] = ((input[:,None,:,i:i+hf,j:j+wf] * grad_output[:,:,None,:,:]).sum((0,3,4))) / batch_size

        pad = (k-1,k-1,k-1,k-1)
        out = F.pad(grad_output, pad, "constant", 0).cuda() #cuda
        weight = torch.flip(weight,[2,3])
        _,_,h0,w0 = out.shape
        for i in range(0,h0-k+1):
            for j in range(0,w0-k+1):
                # print(out[n,:,i:i+hf,j:j+wf].shape, weight[:,c,:,:].shape)
                grad_input[:,:,i,j] = (out[:,:,None,i:i+k,j:j+k] * weight[None,:,:,:,:]).sum((1,3,4))
        #print(grad_input)
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
        global weight1,bias1,weight2,bias2,firstlayer
        #print(self.weight.shape,self.bias.shape)
        if(firstlayer):
           weight1 = self.weight
           bias1 = self.bias
           firstlayer = False
        else:
           weight2 = self.weight
           bias2 = self.bias
           firstlayer = True
        #print("forward",input)
        return convolutionFunction().apply(input, self.weight,self.bias)

class conv2d(nn.Conv2d):
    def __init(self,*kargs,**kwargs):
       super(conv2d, self).__init__()

    def forward(self,input):
        #print("forward - conv2d",input)
        global weight1,bias1,weight2,bias2,firstlayer
        if firstlayer:
           self.bias = bias1
           self.weight = weight1
           firstlayer = False
        else:
           self.bias = bias2
           self.weight = weight2
           firstlayer = True
        #print(input.shape," ",self.weight.shape)
        # self.input = input
        return F.conv2d(input, self.weight,self.bias)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): 
        #print('Input shape',x)
        return self.func(x)

def flatten(x): return x.view(x.shape[0], -1)

def mnist_resize(x): return x.view(-1, 1, 4, 4)

def get_my_cnn_model(data):
    return nn.Sequential(
        Lambda(mnist_resize),
        myconv2d( 1, 2, 2, padding=0,stride=1), nn.ReLU(), #14
        myconv2d( 2,2, 2, padding=0,stride=1), nn.ReLU(), # 7
        # myconv2d(16,32, 3, padding=0,stride=1), nn.ReLU(), # 4
        # myconv2d(32,32, 3, padding=0,stride=1), nn.ReLU(), # 2
        #nn.AdaptiveAvgPool2d(1),
        Lambda(flatten)
        #nn.Linear(8,c)
    )

def get_cnn_model(data):
    return nn.Sequential(
        Lambda(mnist_resize),
        conv2d( 1, 2, 2, padding=0,stride=1), nn.ReLU(), #14
        conv2d( 2,2, 2, padding=0,stride=1), nn.ReLU(), # 7
        #conv2d(16,32, 3, padding=0,stride=1), nn.ReLU(), # 4
        #conv2d(32,32, 3, padding=0,stride=1), nn.ReLU(), # 2
        #nn.AdaptiveAvgPool2d(1),
        Lambda(flatten)
        #nn.Linear(8,c)
    )
def printForward(module,input,output):
    print("input",input)
    print("output",output)
    global ginput,goutput
    ginput,goutput = input,output

def printBackward(module,input,output):
    print("Module",module)
    print("inputgrad",input)
    print("outputgrad",output)
    #print(output.sum(2,3))

def printBackward1(module,input,output):
    print('print brackward1')
    #print("inputgrad",input)
    print("outputgrad",output)
    #print(output.sum(2,3))

device = torch.device('cuda',0)
torch.cuda.set_device(device)

model = get_my_cnn_model(data)
opt = optim.SGD(model.parameters(), lr=0.4)
learn = Learn(model, opt, loss_func, data)
cbfs = [CudaCallback(),AvgStatsCallback(),GradientPrintCallback()] #cuda
run = Runner(learn,cbs = cbfs)
model[1].register_forward_hook(printForward)
model[1].register_backward_hook(printForward)
model[3].register_forward_hook(printForward)
model[3].register_backward_hook(printBackward)
run.fit(1)
#input,output = ginput,goutput
print("****************************** Model1 ***************************")
model1 = get_cnn_model(data)
opt1 = optim.SGD(model1.parameters(), lr=0.4)
learn1 = Learn(model1, opt1, loss_func, data)
run1 = Runner(learn1,cbs = cbfs)
model1[1].register_forward_hook(printForward)
model1[1].register_backward_hook(printForward)
model1[3].register_forward_hook(printForward)
model1[3].register_backward_hook(printBackward)
run1.fit(1)
#input1,output1 = ginput,goutput
'''if torch.equal(input,input1):
   print("Input Same")
else:
   print("input Not Same")

if torch.equal(output,output1):
   print("output Same")
else:
   print("output Not Same")
'''
