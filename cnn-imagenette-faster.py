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
from model import xresnet18
from modelFaster import xresnet_fast18 
import time
from convFaster import *

def load_model(model, state_dict_file_path=None):
    if state_dict_file_path is not None:
        model.load_state_dict(torch.load(state_dict_file_path))
    return model


if __name__ == "__main__":
   start = time.time()
   device = torch.device('cuda',0)
   torch.cuda.set_device(device)
   batch_size = 128
   image_size = 128
   c = 10
   data = load_data(batch_size, image_size,1)
   loss_func = MyCrossEntropy(1)
   loss_func = F.cross_entropy
   #loss_func = MyCrossEntropyFaster(1)   


   #model_big = xresnet18(c_out=c)
   #model_big = load_model(model_big, state_dict_file_path="mymodel/Imagewoof78.pt")
   #model.eval()   

   lr_finder = True
   #model = xresnet_fast18(c_out=c)
   model = xresnet18(c_out=c)
   end = time.time()
   print("Loaded model", end-start)
   
   lr = 0.4
   lr_sched = combine_schedules([0.1, 0.9], [sched_cos(lr/10., lr), sched_cos(lr, lr/1e5)])
   lr_scheduler = ParamScheduler('lr', lr_sched)
   opt = optim.SGD(model.parameters(),lr)
   learn = Learn(model,opt,loss_func, data)
   cbfs = [lr_scheduler,CudaCallback(),AvgStatsCallback()] #cuda
   
   if lr_finder:
      cbfs = [CudaCallback(),LR_find(),Recorder()]

   run = Runner(learn,cbs = cbfs)
   run.fit(80)

#if lr_finder:
#   print(cbfs[-1].lrs)
#   print(cbfs[-1].losses)
