from helper import *
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
from schedulers import combine_schedules, sched_cos
from myconv import myconv2d
from model import xresnet18,xresnet50
from modelFaster import xresnet_fast18,xresnet_fast50 
import time
from convFaster import *

def load_model(model, state_dict_file_path=None):
    if state_dict_file_path is not None:
        model.load_state_dict(torch.load(state_dict_file_path))
    return model

def imagenet_resize(x): return x.view(-1, 3, 128, 128)

if __name__ == "__main__":
   start = time.time()
   device = torch.device('cuda',0)
   torch.cuda.set_device(device)
   batch_size = 64
   image_size = 128
   c = 10
   train = True
   data = load_data(batch_size, image_size,2)
   lr_finder = False
   is_individual_training = True
   
   lr = 0.8
   lr_sched = combine_schedules([0.1, 0.9], [sched_cos(lr/10., lr), sched_cos(lr, lr/1e5)])
   lr_scheduler = ParamScheduler('lr', lr_sched)
   cbfs = [lr_scheduler,CudaCallback()]
 
   if is_individual_training:
      epoch = 80
      compression_factor = 8
      loss_func = F.cross_entropy
      cbfs+=[AvgStatsCallback(metrics=[accuracy,top_k_accuracy])]
      model = xresnet50(c_out=c,resize=imagenet_resize,compression_factor=compression_factor)
   else:
      # currently need to set compression rate manually in convFaster.py
      epoch = 80
      loss_func = MyCrossEntropy(1)
      cbfs+=[AvgStatsCallback()]
      model = xresnet_fast50(c_out=c,resize=imagenet_resize) 

   end = time.time()
   print("Loaded model", end-start)
   
   opt = optim.SGD(model.parameters(),lr)
   learn = Learn(model,opt,loss_func, data)
   
   if lr_finder:
      cbfs = [CudaCallback(),LR_find(),Recorder()]

   run = Runner(learn,cbs = cbfs)
   run.fit(80,train)

#if lr_finder:
#   print(cbfs[-1].lrs)
#   print(cbfs[-1].losses)
