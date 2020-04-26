from helper import get_data_bunch,load_data,load_cifar_data
from run import Runner, Learn
from IPython.core.debugger import set_trace
from torch import tensor, nn, optim
from callback import *
import torch
import torch.nn.functional as F
from mask import randomShape,swastik,star,circle,oval,firstLayerMasking,secondLayerMasking,thirdLayerMasking
from torch.nn.parameter import Parameter
# from .. import init
from torch.autograd import gradcheck
from schedulers import combine_schedules, sched_cos, sched_lin
from myconv import myconv2d
from model import xresnet18,xresnet34,xresnet50
from modelFaster import xresnet_fast18,xresnet_fast34,xresnet_fast50
import time
from convFaster import *

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def flatten(x): return x.view(x.shape[0], -1)

def flatten_parts(x): return x.view(x.shape[0],x.shape[1], -1)

def mnist_resize(x): return x.view(-1, 1, 28, 28)
def cifar_resize(x): return x.view(-1, 3, 32, 32)

def get_cnn_model(data):
    return nn.Sequential(
        Lambda(cifar_resize),
        conv2dFirstLayer( 3, 8, 5, padding=2,stride=2), nn.ReLU(), #14
        conv2dFaster( 8,16, 3, padding=1,stride=2), nn.ReLU(), # 7
        conv2dFaster(16,16, 3, padding=1,stride=2), nn.ReLU(), # 4
        #myconv2d(32,32, 5, padding=1,stride=2,mask=circle(5)), nn.ReLU(), # 2
        #nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        linear(256,c),
    )

def lenet():
    return nn.Sequential(
        Lambda(cifar_resize),
        conv2dFirstLayer( 3, 6, 5, padding=0,stride=1),nn.ReLU(),
        nn.AvgPool2d(2,stride=2),
        conv2dFaster( 6, 16, 5, padding=0,stride=1),nn.ReLU(),
        nn.AvgPool2d(2,stride=2),
        Lambda(flatten),
        linear(400,120),nn.ReLU(),
        linear(120,84),nn.ReLU(),
        linear(84,c),
    )

def conv2(data):
    return nn.Sequential(
        Lambda(cifar_resize),
        conv2dFirstLayer( 3,64,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        conv2dFaster( 64,64,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        Lambda(flatten),
        linear(16384,256),nn.ReLU(),
        linear(256,256),nn.ReLU(),
        linear(256,c)
    )

def conv4(data):
    return nn.Sequential(
        Lambda(cifar_resize),
        conv2dFirstLayer( 3,64,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        conv2dFaster( 64,64,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        conv2dFaster( 64,128,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        conv2dFaster( 128,128,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        Lambda(flatten),
        linear(8192,256),nn.ReLU(),
        linear(256,256),nn.ReLU(),
        linear(256,c)
    )

def conv6(data):
    return nn.Sequential(
        Lambda(cifar_resize),
        conv2dFirstLayer(3,64,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        conv2dFaster( 64,64,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        conv2dFaster( 64,128,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        conv2dFaster( 128,128,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        conv2dFaster( 128,256,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        conv2dFaster( 256,256,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        Lambda(flatten),
        linear(4096,256),nn.ReLU(),
        linear(256,256),nn.ReLU(),
        linear(256,c)
    )


def load_model(model, state_dict_file_path=None):
    if state_dict_file_path is not None:
        model.load_state_dict(torch.load(state_dict_file_path))
    return model

if __name__ == "__main__":
   start = time.time()
   device = torch.device('cuda',0)
   torch.cuda.set_device(device)
   batch_size = 32
   image_size = 32
   c = 100
   train = False
   lr_finder = False
   is_individual_training = False 
   #data = get_data_bunch(batch_size)
   data = load_cifar_data(batch_size, image_size,c)
   
   lr = 0.1
   lr_sched = combine_schedules([0.1, 0.9], [sched_cos(lr/10., lr), sched_cos(lr, lr/1e4)])
   lr_scheduler = ParamScheduler('lr', lr_sched)
   cbfs = [lr_scheduler,CudaCallback()]

   if is_individual_training:
      epoch = 150
      compression_factor = 8
      loss_func = F.cross_entropy
      cbfs+=[AvgStatsCallback(metrics=[accuracy,top_k_accuracy])]
      model = xresnet50(c_out=c,compression_factor=compression_factor)
   else:
      # currently need to set compression rate manually in convFaster.py
      epoch = 200
      loss_func = MyCrossEntropy(1)
      cbfs+=[AvgStatsCallback()]
      model = xresnet_fast50(c_out=c)
      #model = load_model(model, state_dict_file_path="/home/un270/experiments/mymodel/cifar100134.pt")

   #cbfs+=[SaveModelCallback("cifar100_20_1")] 
   #loss_func = MyCrossEntropyFaster(1)
   #model_big = xresnet18(c_out=c)
   #model_big = load_model(model_big, state_dict_file_path="mymodel/cifar100134.pt")

   end = time.time()
   print("Loaded model", end-start)

   opt = optim.SGD(model.parameters(),lr)
   learn = Learn(model,opt,loss_func, data)
   if lr_finder:
      cbfs = [CudaCallback(),LR_find(max_iters=89),Recorder()]

   run = Runner(learn,cbs = cbfs)
   run.fit(epoch)

if lr_finder:
   print(cbfs[-1].lrs)
   print(cbfs[-1].losses)
