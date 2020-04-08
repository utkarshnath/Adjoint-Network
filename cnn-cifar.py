from helper import get_data_bunch,load_data,load_cifar_data
from run import Runner, Learn
from IPython.core.debugger import set_trace
from torch import tensor, nn, optim
from callback import LR_find,Recorder,AvgStatsCallback,CudaCallback,GradientPrintCallback,ParamScheduler
import torch
import torch.nn.functional as F
from mask import *
from torch.nn.parameter import Parameter
# from .. import init
from torch.autograd import gradcheck
from schedulers import combine_schedules, sched_cos
from myconv import myconv2d
from model import xresnet18
import time

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def flatten(x): return x.view(x.shape[0], -1)

def cifar_resize(x): return x.view(-1, 3, 32, 32)
def get_cnn_model(data):
    return nn.Sequential(
        Lambda(cifar_resize),
        myconv2d( 3, 8, 7, padding=2,stride=2), nn.ReLU(), #14
        myconv2d( 8,16, 5, padding=1,stride=2), nn.ReLU(), # 7
        myconv2d(16,16, 5, padding=1,stride=2), nn.ReLU(), # 4
        #myconv2d(32,32, 5, padding=1,stride=2,mask=circle(5)), nn.ReLU(), # 2
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(16,c)
    )

def lenet():
    return nn.Sequential(
        Lambda(cifar_resize),
        myconv2d( 3, 6, 5, padding=0,stride=1),nn.ReLU(),
        nn.AvgPool2d(2,stride=2),
        myconv2d( 6, 16, 5, padding=0,stride=1),nn.ReLU(),
        nn.AvgPool2d(2,stride=2),
        Lambda(flatten),
        nn.Linear(400,120),nn.ReLU(),
        nn.Linear(120,84),nn.ReLU(),
        nn.Linear(84,c)
    )

def conv2(data):
    return nn.Sequential(
        Lambda(cifar_resize),
        myconv2d( 3,64,3, padding=1,stride=1,mask=randomShape(64,3,3,3,0.9)),nn.ReLU(),  # padding = same
        myconv2d( 64,64,3, padding=1,stride=1,mask=randomShape(64,64,3,3,0.9)),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        Lambda(flatten),
        nn.Linear(16384,256),nn.ReLU(),
        nn.Linear(256,256),nn.ReLU(),
        nn.Linear(256,c)
    )

def conv4(data):
    return nn.Sequential(
        Lambda(cifar_resize),
        myconv2d( 3,64,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        myconv2d( 64,64,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        myconv2d( 64,128,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        myconv2d( 128,128,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        Lambda(flatten),
        nn.Linear(8192,256),nn.ReLU(),
        nn.Linear(256,256),nn.ReLU(),
        nn.Linear(256,c)
    )


def conv6(data):
    return nn.Sequential(
        Lambda(cifar_resize),
        myconv2d( 3,64,3, padding=1,stride=1,mask=randomShape(64,3,3,3,0.8)),nn.ReLU(),  # padding = same
        myconv2d( 64,64,3, padding=1,stride=1,mask=randomShape(64,64,3,3,0.8)),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        myconv2d( 64,128,3, padding=1,stride=1,mask=randomShape(128,64,3,3,0.8)),nn.ReLU(),  # padding = same
        myconv2d( 128,128,3, padding=1,stride=1,mask=randomShape(128,128,3,3,0.8)),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        myconv2d( 128,256,3, padding=1,stride=1,mask=randomShape(256,128,3,3,0.8)),nn.ReLU(),  # padding = same
        myconv2d( 256,256,3, padding=1,stride=1,mask=randomShape(256,256,3,3,0.8)),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        Lambda(flatten),
        nn.Linear(4096,256),nn.ReLU(),
        nn.Linear(256,256),nn.ReLU(),
        nn.Linear(256,c)
    )

if __name__ == "__main__":
   start = time.time()
   device = torch.device('cuda',0)
   torch.cuda.set_device(device)
   batch_size = 32
   image_size = 32
   c = 100
   data = load_cifar_data(batch_size, image_size,100)
   loss_func = F.cross_entropy
   lr_finder = False
   mask = randomShape1(3,3,0.8)
   print(mask)
   #a = torch.zeros(3,3).cuda()
   #a[0][1] = a[1][0] = 1
   #a[0][1] = a[1][0] = a[1][2] = a[2][1] = 1
   #a[0][1] = a[1][2] = 0
   #print(a)
   model = xresnet18(mask)
   end = time.time()
   print("Loaded model", end-start)
   lr = 0.1
   lr_sched = combine_schedules([0.1, 0.9], [sched_cos(lr/10., lr), sched_cos(lr, lr/1e5)])
   lr_scheduler = ParamScheduler('lr', lr_sched)

   opt = optim.SGD(model.parameters(),lr)
   learn = Learn(model,opt,loss_func, data)
   cbfs = [lr_scheduler,CudaCallback(),AvgStatsCallback()] #cuda
   if lr_finder:
      cbfs = [CudaCallback(),LR_find(max_iters=40),Recorder()]

   run = Runner(learn,cbs = cbfs)
   run.fit(100)
   #print(model[1].weight)
   #print()
   #print(model[3].weight)
   #print()
   #print(model[5].weight)
   #print(model.layer[2].weight)

if lr_finder:
   print(cbfs[-1].lrs)
   print(cbfs[-1].losses)
