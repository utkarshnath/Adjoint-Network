from helper import *
from run import Runner, Learn
from IPython.core.debugger import set_trace
from torch import tensor, nn, optim
from callback import *
import torch
import torch.nn.functional as F
from mask import *
from optimizers import *
from torch.nn.parameter import Parameter
# from .. import init
from torch.autograd import gradcheck
from schedulers import combine_schedules, sched_cos
from myconv import myconv2d
from model import *
import time
from modelFaster import xresnet_fast18,xresnet_fast34,xresnet_fast50

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

def conv6_adjoint(compression_factor,masking_factor):
    return nn.Sequential(
        Lambda(cifar_resize),
        conv2dFirstLayer(3,64,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        conv2dFaster( 64,64,3, padding=1,stride=1,mask_layer=False),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        conv2dFaster( 64,128,3, padding=1,stride=1,mask_layer=False,compression_factor=compression_factor,masking_factor=masking_factor),nn.ReLU(),  # padding = same
        conv2dFaster( 128,128,3, padding=1,stride=1,mask_layer=False,compression_factor=compression_factor,masking_factor=masking_factor),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        conv2dFaster( 128,256,3, padding=1,stride=1,mask_layer=True,compression_factor=compression_factor,masking_factor=masking_factor),nn.ReLU(),  # padding = same
        conv2dFaster( 256,256,3, padding=1,stride=1,mask_layer=True,compression_factor=compression_factor,masking_factor=masking_factor),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        Lambda(flatten),
        linear(4096,256),nn.ReLU(),
        linear(256,256),nn.ReLU(),
        linear(256,c)
    )


def conv6_individual(data):
    return nn.Sequential(
        Lambda(cifar_resize),
        nn.Conv2d( 3,64,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        nn.Conv2d( 64,64,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        nn.Conv2d( 64,128,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        nn.Conv2d( 128,128,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        nn.Conv2d( 128,256,3, padding=1,stride=1),nn.ReLU(),  # padding = same
        nn.Conv2d( 256,256,3, padding=1,stride=1),nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        Lambda(flatten),
        nn.Linear(4096,256),nn.ReLU(),
        nn.Linear(256,256),nn.ReLU(),
        nn.Linear(256,c)
    )

def imagenet_resize(x): return x.view(-1, 3, 224, 224)


if __name__ == "__main__":
   start = time.time()
   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   batch_size = 64
   image_size = 224
   c = 37
   train = False
   lr_finder = False
   compression_factor = 8
   masking_factor = 0.5
   is_individual_training = True
   #data = get_data_bunch(batch_size)
   data = load_fastai_data(batch_size, image_size)

   lr = 1e-3
   lr_sched = combine_schedules([0.1, 0.9], [sched_cos(lr/10., lr), sched_cos(lr, lr/1e5)])
   beta1_sched = combine_schedules([0.1, 0.9], [sched_cos(0.95, 0.85), sched_cos(0.85, 0.95)])
   lr_scheduler = ParamScheduler('lr', lr_sched)
   beta1_scheduler = ParamScheduler('beta1', beta1_sched)
   cbfs = [lr_scheduler,beta1_scheduler,CudaCallback()]

   if is_individual_training:
      epoch = 100
      compression_factor = 1
      loss_func = F.cross_entropy
      cbfs+=[AvgStatsCallback(metrics=[accuracy,top_k_accuracy])]
      #model = conv6_individual(data)
      model = xresnet50(c_out=c,resize=imagenet_resize,compression_factor=compression_factor)
   else:
      # currently need to set compression rate manually in convFaster.py
      epoch = 100
      loss_func = MyCrossEntropy(0)
      cbfs+=[AvgStatsCallback(),lossScheduler()]
      #model = conv6(data)
      model = xresnet_fast50(c_out=c,resize=imagenet_resize ,compression_factor=compression_factor, masking_factor=masking_factor)

   end = time.time()
   print("Loaded model", end-start)


   opt = StatefulOptimizer(model.parameters(), [weight_decay, adam_step],stats=[AverageGrad(), AverageSqGrad(), StepCount()], lr=0.001, wd=1e-2, beta1=0.9, beta2=0.99, eps=1e-6)
   learn = Learn(model,opt,loss_func, data)
   if lr_finder:
      cbfs = [CudaCallback(),LR_find(max_iters=89),Recorder()]

   run = Runner(learn,cbs = cbfs)
   run.fit(epoch)
