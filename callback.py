import time, datetime
import torch
from run import CancelTrainException, CancelEpochException, CancelBatchException


class CallBacks():
    _order = 0
    def set_runner(self, run): 
        self.run=run

    def __getattr__(self, k): 
        return getattr(self.run, k, None)


def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()    
def top_k_accuracy(out, yb, k=5):
    idx = out.topk(k=k, dim=1)[1]
    yb = yb.unsqueeze(dim=1).expand_as(idx)
    return (yb == idx).max(dim=1)[0].float().mean()

class CudaCallback(CallBacks):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()

class GradientPrintCallback(CallBacks):
    def after_batch(self):
        pass  # print("callback called")
        #print('weight',self.model[1].weight,'\n')
        #print('bias', self.model[1].bias,'\n')
        #print('weight grad',self.model[1].weight.grad,'\n')
        #self.learn.opt.zero_grad()
        #if self.iter >= 1 : raise CancelTrainException()
                         
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

    
# should probably run at the end of other call backs
class AvgStatsCallback(CallBacks):
    _order = 50
   
    def __init__(self, metrics=[accuracy]):
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
        for o in [self.train_stats,  self.valid_stats]:
            stats += [f'{v:.6f}' for v in o.avg_stats]
        
        t = self.valid_start_time if self.train_start_time is None else self.train_start_time 
        stats += [str(datetime.timedelta(seconds=int(time.time()-t))) ]
        self.logger(stats)


