import time, datetime
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from adjointNetworkNAS import *
from run import CancelTrainException, CancelEpochException, CancelBatchException
import matplotlib.pyplot as plt

class CallBacks():
    _order = 0
    def set_runner(self, run): 
        self.run=run

    def __getattr__(self, k): 
        return getattr(self.run, k, None)


def accuracy(out, yb):
    l,_ = out.shape
    return (torch.argmax(out, dim=1)==yb).float().mean()

def top_k_accuracy(out, yb, k=5):
    l,_ = out.shape
    idx = out.topk(k=k, dim=1)[1]
    yb = yb.unsqueeze(dim=1).expand_as(idx)
    return (yb == idx).max(dim=1)[0].float().mean()


def accuracy_faster(out, yb):
    l,_ = out.shape
    return (torch.argmax(out[:l//2], dim=1)==yb).float().mean()
def accuracy1_faster(out, yb):
    l,_ = out.shape
    return (torch.argmax(out[l//2:], dim=1)==yb).float().mean()

def top_k_accuracy_faster(out, yb, k=5):
    l,_ = out.shape
    idx = out[:l//2].topk(k=k, dim=1)[1]
    yb = yb.unsqueeze(dim=1).expand_as(idx)
    return (yb == idx).max(dim=1)[0].float().mean()

def top_k_accuracy1_faster(out, yb, k=5):
    l,_ = out.shape
    idx = out[l//2:].topk(k=k, dim=1)[1]
    yb = yb.unsqueeze(dim=1).expand_as(idx)
    return (yb == idx).max(dim=1)[0].float().mean()

def nll(out, yb):
    l,_ = out.shape
    log_preds = F.log_softmax(out[:l//2], dim=-1)
    nll1 = F.nll_loss(log_preds, yb)
    return nll1

class CudaCallback(CallBacks):
    def __init__(self,device):
        self.device = device

    def begin_fit(self):
        self.model = self.model.cuda()
    
    def begin_batch(self): self.run.xb,self.run.yb = self.xb.to(self.device),self.yb.to(self.device)

class lossScheduler(CallBacks):
    def after_epoch(self):
        x = self.epoch/self.epochs
        self.learn.loss_func = AdjointLoss(min(4*(x**2),1))
           
           
                         
class Stats():
    def __init__(self, metrics, in_train):
        self.metrics = metrics
        self.in_train = in_train
        self.reset()

    def reset(self):
        self.tol_loss = torch.tensor([0.0])
        self.tol_metrics = [0.]*len(self.metrics)
        self.count = 0

    @property
    def all_stats(self): return [self.tol_loss.item()] + self.tol_metrics
    @property
    def avg_stats(self): return [s/self.count if self.count !=0 else -1.0 for s in self.all_stats]

    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        batch_size = run.xb.shape[0]

        self.tol_loss += batch_size * run.loss.cpu()
        for i, metric in enumerate(self.metrics):
            self.tol_metrics[i] += batch_size * metric(run.pred, run.yb)
        self.count += batch_size

class ParamScheduler(CallBacks):
    _order = 5
    
    def __init__(self, pname, sched_func, using_torch_optim=False):
        self.pname = pname
        self.sched_func = sched_func 
        self.iter = 0.0
        self.using_torch_optim = using_torch_optim

    def set_param(self):
        self.iter += 1.0/self.iters

        hypers = self.opt.param_groups       
        if not self.using_torch_optim:
            hypers = self.opt.hypers 
       
        for h in hypers:
            h[self.pname] = self.sched_func((self.iter+self.start_epoch)/self.epochs)
   
    def begin_batch(self): 
        if self.in_train: self.set_param()


# probably run after cuda 
class LR_find(CallBacks):
    _order = 1
  
    def __init__(self, max_iters=100, using_torch_optim=True):
        self.using_torch_optim = using_torch_optim
        self.max_iters = max_iters
 
    def begin_fit(self, min_lr=1e-6, max_lr=10, tol_factor=10):
        self.best_loss, self.tol_factor = (1e9, 0), tol_factor
        self.min_lr, self.max_lr = min_lr, max_lr

    def begin_batch(self):
        if not self.in_train: return

        pos = self.iter/self.max_iters
        self.curr_lr = self.min_lr * (self.max_lr/self.min_lr) ** pos

        hypers = self.opt.param_groups       
        if not self.using_torch_optim:
            hypers = self.opt.hypers 

        for h in hypers:
            h['lr'] = self.curr_lr
        
        if self.iter >= self.max_iters:
           print("##################") 
           raise CancelTrainException()

    def after_loss(self):
        if not self.in_train: return
        if self.loss < self.best_loss[0]:
            self.best_loss = (self.loss, self.iter, self.curr_lr)
            print(self.best_loss)

        if self.loss >= self.tol_factor*self.best_loss[0]:
            raise CancelTrainException()

    def after_cancel_train(self):
        print("Best loss value = {} at iteration = {}".format(self.best_loss[0], self.best_loss[1])) 
        print("The learning rate was = {}".format(self.best_loss[2]))


class Recorder(CallBacks):
    _order = 20
    
    def __init__(self, using_torch_optim=True):
        self.lrs = []
        self.losses = []
        self.using_torch_optim = using_torch_optim

    def after_loss(self):
        if not self.in_train: return
        self.losses.append(self.loss.detach().cpu())

    def after_batch(self):
        if not self.in_train: return
        
        if self.using_torch_optim:
            lr = self.opt.param_groups[-1]['lr'] 
        else: 
            lr = self.opt.hypers[-1]['lr']
        self.lrs.append(lr)

    def plot_lr(self): 
        plt.plot(self.lrs)
        plt.show()

    def plot_loss(self):
        plt.plot(self.losses)
        plt.show()

class SaveModelCallback(CallBacks):
    def __init__(self,name,save_dir="/scratch/un270/model/"):
        model_directory = os.path.join(save_dir,name)
        self.name = name
        if not os.path.isdir(model_directory):
            os.mkdir(model_directory)   
        self.model_directory = model_directory

    def after_epoch(self):
        curr_name = os.path.join(self.model_directory, str(self.epoch) + ".pt")
        torch.save(self.learn.model.state_dict(), curr_name) 

class NormalizeCallback(CallBacks):
    _order = 2
    def __init__(self,device,oncuda=True):
        self.mean = tensor([0.485, 0.456, 0.406])
        self.std = tensor([0.229, 0.224, 0.225])
        if oncuda: self.mean, self.std = self.mean.to(device), self.std.to(device)

    def begin_batch(self): 
        self.run.xb = (self.xb - self.mean[...,None,None]) / self.std[...,None,None]

# Should run after stats callback
class InferenceCallback(CallBacks):
    _order = 51 

    def begin_epoch(self):
        if self.in_train: raise CancelEpochException()
 
    def after_epoch(self): raise CancelTrainException()

# should probably run at the end of other call backs
class AvgStatsCallback(CallBacks):
    _order = 50
   
    def __init__(self, metrics=[nll,accuracy_faster,top_k_accuracy_faster,accuracy1_faster,top_k_accuracy1_faster]):
        self.train_stats = Stats(metrics, True)
        self.valid_stats = Stats(metrics, False)    
        self.train_start_time, self.valid_start_time = None, None

    def begin_epoch(self):
        if self.in_train:
            self.train_stats.reset()
            self.train_start_time = time.time()
        else:
            self.valid_stats.reset()
            self.valid_start_time = time.time()

    def begin_batch(self):
        self.batch_size = self.xb.shape[0]

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)

    def after_epoch(self):
        stats = ["{}/{}".format(self.epoch+1, self.epochs)]
        for o in [self.train_stats ,self.valid_stats]:
            stats += [f'{v:.6f}' for v in o.avg_stats]
        
        t = self.valid_start_time if self.train_start_time is None else self.train_start_time 
        stats += [str(datetime.timedelta(seconds=int(time.time()-t))) ]
        self.logger(stats)

class OpTime():
    def __init__(self): self.reset() 
    def reset(self): self.start = time.time()    
    def compute_elasped_time(self): return time.time() - self.start

# times the different operations.
# should run before everything else
class DebugTimeCallback(CallBacks):
    _order = -1
    
    def __init__(self, print_freq=50, max_iters=100):
        self.print_freq = print_freq
        self.max_iters = max_iters 
        self.reset_times()

    def print_times(self):
        times = [self.data_loading_time, self.model_compute_time, self.loss_compute_time, self.backward_compute_time, self.opt_step_time]
        avg_t = [t/self.curr_counter for t in times]
        print(avg_t)
        print(sum(times)/self.curr_counter)        

    def reset_times(self):
        self.data_loading_time, self.model_compute_time = 0, 0
        self.loss_compute_time, self.backward_compute_time = 0, 0
        self.opt_step_time = 0
        self.curr_counter = 0

    def compute_elasped_time_and_reset(self, fr_time, crr_time):
        ret_time = 0
        if fr_time is not None: 
            ret_time = fr_time.compute_elasped_time()
        if crr_time is None: crr_time = OpTime()
        else: crr_time.reset()

        return ret_time, crr_time

    def begin_batch(self):
        ret_time, crr_time = self.compute_elasped_time_and_reset(self.after_batch_time, self.begin_batch_time)
        self.data_loading_time += ret_time
        self.begin_batch_time = crr_time

    def after_pred(self):
        ret_time, crr_time = self.compute_elasped_time_and_reset(self.begin_batch_time, self.after_pred_time)
        self.model_compute_time += ret_time
        self.after_pred_time = crr_time

    def after_loss(self):
        ret_time, crr_time = self.compute_elasped_time_and_reset(self.after_pred_time, self.after_loss_time)
        self.loss_compute_time += ret_time
        self.after_loss_time = crr_time

    def before_step(self):
        ret_time, crr_time = self.compute_elasped_time_and_reset(self.after_loss_time, self.before_step_time)
        self.backward_compute_time += ret_time
        self.before_step_time = crr_time

    def before_zero_grad(self):
        ret_time, crr_time = self.compute_elasped_time_and_reset(self.before_step_time, self.before_zero_grad_time)
        self.opt_step_time += ret_time
        self.before_zero_grad_time = crr_time

    def after_batch(self):
        self.after_batch_time = OpTime()
        self.curr_counter += 1

        if self.iter % self.print_freq == 0:
            self.print_times()
            self.reset_times()
        
        if self.iter >= self.max_iters: raise CancelTrainException()

