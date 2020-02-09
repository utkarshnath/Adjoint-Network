from helper import get_data_bunch
from run import Runner, Learn
from IPython.core.debugger import set_trace
import torch.nn.functional as F
from torch import tensor, nn, optim
from callback import AvgStatsCallback,CudaCallback
from torch.autograd import Function
import torch
from torch.nn.parameter import Parameter 

batch_size = 2
c = 10
data = get_data_bunch(batch_size)
loss_func = F.cross_entropy

class convolutionFunction(Function):
    def __init__(self,in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = torch.Tensor(out_channels,in_channels,kernel_size,kernel_size)

    @staticmethod
    def forward(context,input,weight,bias):
        # out = nn.functional.conv2d(input,weight,bias)
        context.save_for_backward(input,weight,bias)
        # weight = torch.Tensor(8,1,kernel_size,kernel_size)
        # print(weight.shape)
        N,C,h,w = input.shape
        out_channels,_,hf,wf = weight.shape
        out = torch.Tensor(N,out_channels,h-hf+1,w-wf+1).cuda()
        # h = 10, w = 10, hf = self.kernel_size, wf = self.kernel_size
        for n in range(0,N):
            # print(n)
            for c in range(0,out_channels):
                for i in range(0,h-hf+1):
                    for j in range(0,w-wf+1):
                        # print(i,i+hf)
                        # print(input[n][:,i:i+hf,j:j+wf].shape,weight[c][:][:][:].shape)
                        out[n][c][i][j] = (input[n][:,i:i+hf,j:j+wf] * weight[c][:][:][:]).sum()
        return out
	
    @staticmethod
    def backward(context,grad_output):
        # print('backward called',grad_output.shape)
        input,weight,bias = context.saved_tensors
        grad_input, grad_weight = torch.Tensor(input.shape).cuda(),torch.Tensor(weight.shape).cuda()
        out_channels,in_channels,k,_ = grad_weight.shape
        _,_,hf,wf = grad_output.shape
        for f in range(0,out_channels):
            for c in range(0,in_channels):
                for i in range(0,k):
                    for j in range(0,k):
                        # print(input[:,c,i:i+hf,j:j+wf].shape, grad_output[:,f,:,:].shape)
                        grad_weight[f][c][i][j] = (input[:,c,i:i+hf,j:j+wf] * grad_output[:,f,:,:]).sum()
        
        pad = (k-1,k-1,k-1,k-1)
        out = F.pad(grad_output, pad, "constant", 0).cuda()
        torch.flip(weight,[2,3])
        _,_,h0,w0 = out.shape
        for n in range(0,input.shape[0]):
            for c in range(0,in_channels):
                for i in range(0,h0-k+1):
                    for j in range(0,w0-k+1):
                        # print(out[n,:,i:i+hf,j:j+wf].shape, weight[:,c,:,:].shape)
                        grad_input[n][c][i][j] = (out[n,:,i:i+k,j:j+k] * weight[:,c,:,:]).sum() 
        return grad_input,grad_weight,bias

class myconv2d(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,*kargs,**kwargs):
        super(myconv2d, self).__init__(in_channels,out_channels,kernel_size,*kargs, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size).cuda())

    def forward(self,input):
        return convolutionFunction(1,8,5).apply(input, self.weight, self.bias)
    

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

opt = optim.SGD(model.parameters(), lr=0.4)
learn = Learn(model, opt, loss_func, data)
cbfs = [CudaCallback(),AvgStatsCallback()]
run = Runner(learn,cbs = cbfs)

run.fit(5)
