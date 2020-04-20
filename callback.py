import time, datetime
import torch
import os
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
    return (torch.argmax(out[:l], dim=1)==yb).float().mean()
def accuracy1(out, yb):
    l,_ = out.shape
    return (torch.argmax(out[l//2:], dim=1)==yb).float().mean()
def accuracy2(out, yb):
    l,_ = out.shape
    return (torch.argmax(out[l//2:3*l//4], dim=1)==yb).float().mean()
def accuracy3(out, yb):
    l,_ = out.shape
    return (torch.argmax(out[3*l//4:], dim=1)==yb).float().mean()    

def top_k_accuracy(out, yb, k=5):
    l,_ = out.shape
    idx = out[:l].topk(k=k, dim=1)[1]
    yb = yb.unsqueeze(dim=1).expand_as(idx)
    return (yb == idx).max(dim=1)[0].float().mean()

def top_k_accuracy1(out, yb, k=5):
    l,_ = out.shape
    idx = out[l//2:].topk(k=k, dim=1)[1]
    yb = yb.unsqueeze(dim=1).expand_as(idx)
    return (yb == idx).max(dim=1)[0].float().mean()

def top_k_accuracy2(out, yb, k=5):
    l,_ = out.shape
    idx = out[l//2:3*l//4].topk(k=k, dim=1)[1]
    yb = yb.unsqueeze(dim=1).expand_as(idx)
    return (yb == idx).max(dim=1)[0].float().mean()

def top_k_accuracy3(out, yb, k=5):
    l,_ = out.shape
    idx = out[3*l//4:].topk(k=k, dim=1)[1]
    yb = yb.unsqueeze(dim=1).expand_as(idx)
    return (yb == idx).max(dim=1)[0].float().mean()


class CudaCallback(CallBacks):
    def begin_fit(self):
        self.model.cuda()
        #self.model_big.cuda()
    def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()

class GradientPrintCallback(CallBacks):
    def after_batch(self):
        #pass  # print("callback called")
        #print('weight',self.model[1].weight,'\n')
        #print('bias', self.model[1].bias,'\n')
        #print('weight grad',self.model[1].weight.grad,'\n')
        self.learn.opt.zero_grad()
        if self.iter >= 1 : raise CancelTrainException()
                         
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

        self.tol_loss += batch_size * run.loss
        for i, metric in enumerate(self.metrics):
            self.tol_metrics[i] += batch_size * metric(run.pred, run.yb)
        self.count += batch_size

# should run after cuda but before everything else
class ParamScheduler(CallBacks):
    _order = 5
    
    def __init__(self, pname, sched_func, using_torch_optim=False):
        self.pname = pname
        self.sched_func = sched_func 
        self.iter = 0.0


    def set_param(self):
        self.iter += 1.0/self.iters
        for p in self.opt.param_groups:
            p[self.pname] = self.sched_func((self.iter+self.start_epoch)/self.epochs)
   
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
    def __init__(self,name,save_dir="/scratch/un270/"):
        model_directory = os.path.join(save_dir, "models")
        self.name = name
        if not os.path.isdir(model_directory):
            os.mkdir(model_directory)   
        self.model_directory = model_directory

    def after_epoch(self):
        if self.epoch>50:
           curr_name = os.path.join(self.model_directory, self.name + str(self.epoch)+".pt")
           torch.save(self.learn.model.state_dict(), curr_name) 

# should probably run at the end of other call backs
class AvgStatsCallback(CallBacks):
    _order = 50
   
    def __init__(self, metrics=[accuracy,top_k_accuracy]):
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


