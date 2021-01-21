from helper import *
from run import Runner, Learn
from IPython.core.debugger import set_trace
from torch import tensor, nn, optim
from callback import *
import torch
import torch.nn.functional as F
from mask import randomShape,swastik,star,circle,oval,firstLayerMasking,secondLayerMasking,thirdLayerMasking
from torch.nn.parameter import Parameter
from functools import partial
from torch.autograd import gradcheck
from schedulers import combine_schedules, sched_cos
from myconv import myconv2d
from model import xresnet18,xresnet50,xresnet101
from modelNAS import xresnet_fast18,xresnet_fast50,xresnet_fast50X2,xresnet_fast101
import time
from adjointNetworkNAS import *
from optimizers import *
import argparse
from config import *

parser = argparse.ArgumentParser(description='Adjoint Network')
parser.add_argument('--lr', type=int, default=0.001, help='')
parser.add_argument('--is_sgd', type=str, default='False', help='')
parser.add_argument('--is_adjoint_training', type=str, default='True',help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--compression_factor', type=int, default=4, help='')
parser.add_argument('--masking_factor', type=float, default=None, help='')
parser.add_argument('--image_size', type=int, default=32, help='')
parser.add_argument('--classes', type=int, default=100, help='')
parser.add_argument("--epoch", type=int, default=100, help="")
parser.add_argument("--resnet", type=int, default=50, help="")
parser.add_argument("--dataset", type=str, default='cifar100', help="")
parser.add_argument("--default_config", type=str, default='True', help="")
args = parser.parse_args()

def load_model(model, state_dict_file_path=None):
    if state_dict_file_path is not None:
        model.load_state_dict(torch.load(state_dict_file_path))
    return model

def load_searched_model(model, path):
    state_dict = torch.load(path)
    for k,v in state_dict.items():
        if(k.find("gumbel")!=-1):
           model.state_dict()[k].copy_(v)

def dataset_resize(image_size,x): return x.view(-1, 3, image_size, image_size)


if __name__ == "__main__":
   start = time.time()
   device = torch.device('cuda',0)
   # torch.cuda.set_device(device)
   if args.default_config=='True':
     batch_size, image_size, lr, c, epoch, is_sgd =  get_default_config(args.dataset)
   else:
     batch_size = args.batch_size
     image_size = args.image_size
     c = args.classes
     epoch = args.epoch
     is_sgd = False if args.is_sgd=='False' else True
     lr = args.lr
   
   if args.dataset=='imagenet':
     data = load_data(batch_size, image_size,0)
   elif args.dataset=='imagewoof':
     data = load_data(batch_size, image_size,2)
   elif args.dataset=='cifar100':
     data = load_cifar_data(batch_size, image_size,100)
   elif args.dataset=='cifar10':
     data = load_cifar_data(batch_size, image_size,10)
   elif args.dataset=='pets':
     data = load_fastai_data(batch_size, image_size)

   data_resize = partial(dataset_resize,image_size)
   is_individual_training = False if args.is_adjoint_training=='True' else True
   last_epoch_done_idx = None
   compression_factor = args.compression_factor
   masking_factor = args.masking_factor
   is_student_teacher = False
   architecture_search = False

   print('************* Current Settings **********')
   print('dataset',args.dataset)
   print('batch_size',batch_size)
   print('image_size',image_size)
   print('lr',lr)
   print('c',c)
   print('epoch',epoch)
   print('is sgd',is_sgd)
   print('is_individual_training',is_individual_training)
   print('compression_factor',compression_factor)
   print('masking_factor',masking_factor)
   print('resnet',args.resnet)
   print('is_student_teacher',is_student_teacher)
   print('architecture_search', architecture_search)
   print('*****************************************')

   if is_sgd:
      lr_sched = combine_schedules([0.1, 0.9], [sched_cos(lr/10., lr), sched_cos(lr, lr/1e5)])
      lr_scheduler = ParamScheduler('lr', lr_sched,using_torch_optim=True)
      cbfs = [NormalizeCallback(device),lr_scheduler,CudaCallback()]
   else:
      lr_sched = combine_schedules([0.1, 0.9], [sched_cos(lr/10., lr), sched_cos(lr, lr/1e5)])
      beta1_sched = combine_schedules([0.1, 0.9], [sched_cos(0.95, 0.85), sched_cos(0.85, 0.95)])
      lr_scheduler = ParamScheduler('lr', lr_sched)
      beta1_scheduler = ParamScheduler('beta1', beta1_sched)
      cbfs = [NormalizeCallback(device),lr_scheduler,beta1_scheduler,CudaCallback(device)]

   if is_individual_training:
      loss_func = F.cross_entropy
      cbfs+=[AvgStatsCallback(metrics=[accuracy,top_k_accuracy])]
      resnet = args.resnet
      if resnet==18:
         model = xresnet18(c_out=c,resize=data_resize,compression_factor=compression_factor)
      elif resnet==34:
          model = xresnet34(c_out=c,resize=data_resize,compression_factor=compression_factor)
      elif resnet==50:
          model = xresnet50(c_out=c,resize=data_resize,compression_factor=compression_factor)
      elif resnet==101:
         model = xresnet101(c_out=c,resize=data_resize,compression_factor=compression_factor)
      elif resnet==152:
          model = xresnet152(c_out=c,resize=data_resize,compression_factor=compression_factor)
      else:
         print("Resnet model supported are 18, 34, 50, 101, 152")
      if last_epoch_done_idx is not None: model = load_model(model, state_dict_file_path="/scratch/un270/model/resnet50-ind4-imagenet/{}.pt".format(last_epoch_done_idx))
   else:
      loss_func = AdjointLoss(0)
      cbfs+=[AvgStatsCallback()]
      resnet = args.resnet
      if resnet==18:
         model = xresnet_fast18(c_out=c, resize=data_resize, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor)
      elif resnet==34:
         model = xresnet_fast34(c_out=c, resize=data_resize, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor)
      elif resnet==50:
         model = xresnet_fast50(c_out=c, resize=data_resize, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor)
      elif resnet==250:
         model = xresnet_fast50X2(c_out=c, resize=data_resize, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor)
      elif resnet==101:
         model = xresnet_fast101(c_out=c, resize=data_resize, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor)
      elif resnet==152:
         model = xresnet_fast152(c_out=c, resize=data_resize, architecture_search=architecture_search, compression_factor=compression_factor, masking_factor=masking_factor)
      else:
         print("Resnet model supported are 18, 34, 50, 101, 152")
      if last_epoch_done_idx is not None: model = load_model(model, state_dict_file_path="/scratch/un270/model/Adj-resnet50-imagenet-60epoch-adam/{}.pt".format(last_epoch_done_idx))
  
   #cbfs+=[SaveModelCallback('newlat-t15-schedular-1e-13')]
   # model = nn.DataParallel(model)
   # model = model.to(device)
   
   if architecture_search == False:
       load_searched_model(model, "/scratch/un270/model/Adjoint-Experiments/Nas/updated_config/search_cifar24816_e16_bs32/145.pt")
   else:
       cbfs+=[SaveModelCallback('search_cifar24816_e16_bs32')]

   cbfs += [lossScheduler()]

   #type 1,2
   #loss_func = AdjointLoss(0)
   #cbfs += [lossScheduler()]

   teacher_model = None
   if is_student_teacher:
      teacher_model = xresnet18(c_out=c,resize=data_resize)
      teacher_model = load_model(teacher_model, state_dict_file_path='/scratch/un270/model/{}-individual-18/{}.pt'.format(args.dataset,epoch-1))
      teacher_model.cuda()
      loss_func = TeacherStudentLoss()

   end = time.time()
   print("Loaded model", end-start)

   if is_sgd:
      opt = optim.SGD(model.parameters(),lr)
   else:
      opt = StatefulOptimizer(model.parameters(), [weight_decay, adam_step],stats=[AverageGrad(), AverageSqGrad(), StepCount()], lr=0.001, wd=1e-2, beta1=0.9, beta2=0.99, eps=1e-6)

   learn = Learn(model,opt,loss_func, data,teacher_model,architecture_search)
   

   run = Runner(learn,cbs = cbfs)
   run.fit(epoch, start_epoch = 0 if last_epoch_done_idx is None else last_epoch_done_idx+1)
