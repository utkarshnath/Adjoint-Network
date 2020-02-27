from helper import get_data_bunch
from run import Runner, Learn
from IPython.core.debugger import set_trace
from torch import tensor, nn, optim
from callback import LR_find,Recorder,AvgStatsCallback,CudaCallback,GradientPrintCallback,ParamScheduler
import torch
import torch.nn.functional as F
from mask import randomShape,swastik,star,circle,oval,firstLayerMasking,secondLayerMasking,thirdLayerMasking
from torch.nn.parameter import Parameter 
# from .. import init
from torch.autograd import gradcheck
from schedulers import combine_schedules, sched_cos
from myconv import myconv2d

batch_size = 512
c = 10
data = get_data_bunch(batch_size)
loss_func = F.cross_entropy


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
        myconv2d( 1, 8, 7, padding=2,stride=2,mask=firstLayerMasking(7,8)), nn.ReLU(), #14
        myconv2d( 8,16, 5, padding=1,stride=2,mask=secondLayerMasking(5,16)), nn.ReLU(), # 7
        myconv2d(16,16, 5, padding=1,stride=2,mask=thirdLayerMasking(5,16)), nn.ReLU(), # 4
        #myconv2d(32,32, 5, padding=1,stride=2,mask=circle(5)), nn.ReLU(), # 2
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(16,c)
    )

if __name__ == "__main__":
   device = torch.device('cuda',0)
   torch.cuda.set_device(device)
   lr_finder = False
   model = get_cnn_model(data)
   print("Random masking")

   lr = 0.6
   lr_sched = combine_schedules([0.1, 0.9], [sched_cos(lr/10., lr), sched_cos(lr, lr/1e4)])
   lr_scheduler = ParamScheduler('lr', lr_sched)

   opt = optim.SGD(model.parameters(),lr)
   learn = Learn(model,opt,loss_func, data)
   cbfs = [CudaCallback(),AvgStatsCallback(),lr_scheduler] #cuda
   if lr_finder:
      cbfs = [CudaCallback(),LR_find(max_iters=45),Recorder()]

   run = Runner(learn,cbs = cbfs)
   run.fit(50)

if lr_finder:
   print(cbfs[-1].lrs) 
   print(cbfs[-1].losses)


