import torch
import time

class DataBunch():
    def __init__(self, train_dl, valid_dl):
        self.train_dl = train_dl
        self.valid_dl = valid_dl

    @property
    def train_ds(self): 
        return self.train_dl.dataset
        
    @property
    def valid_ds(self): 
        return self.valid_dl.dataset


class Learn():
    def __init__(self, model, opt, loss_func, data,ngpu):
        self.model, self.opt = model, opt 
        self.loss_func, self.data = loss_func, data
        self.ngpu = ngpu


class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

def updateSequenceOutput(output,ngpu=4):
    l = output.shape[0]
    factor = ngpu*2
    if ngpu==1:
        return output
    elif ngpu==3:
        first = torch.cat([output[:l//factor], output[2*l//factor:3*l//factor],output[4*l//factor:5*l//factor]], dim=0)
        second = torch.cat([output[l//factor:2*l//factor], output[3*l//factor:4*l//factor],output[5*l//factor:]], dim=0)
    elif ngpu==4:
        first = torch.cat([output[:l//factor], output[2*l//factor:3*l//factor],output[4*l//factor:5*l//factor],output[6*l//factor:7*l//factor]], dim=0)
        second = torch.cat([output[l//factor:2*l//factor], output[3*l//factor:4*l//factor],output[5*l//factor:6*l//factor],output[7*l//factor:]], dim=0)
    return torch.cat([first,second],dim=0) 

class Runner():
    def __init__(self, learn, cbs=None):
        self.learn = learn
        if cbs is not None:
            self.cbs = cbs    
            for cb in cbs:
                cb.set_runner(self)  

        self.logger = print    
    
    @property
    def opt(self): return self.learn.opt

    @property
    def model(self): return self.learn.model

    @property
    def loss_func(self): return self.learn.loss_func

    @property
    def data(self): return self.learn.data
 
    def one_batch(self, xb, yb, *args):
        try:
            self.iter += 1
            self.xb, self.yb = xb, yb
            self.args = args
             
            self.handle("begin_batch")
            self.pred = self.learn.model(self.xb)
            self.pred = updateSequenceOutput(self.pred,self.learn.ngpu) 
            
            self.handle("after_pred")                
            
            self.loss = self.learn.loss_func(self.pred,self.yb)
            #print(self.loss)
            self.handle("after_loss")
            
            if not self.in_train: return 
            
            self.handle("before_backward")                
            self.loss.backward()
            
            self.handle("before_step") 
            self.learn.opt.step()    
            
            self.handle("before_zero_grad")
            self.learn.opt.zero_grad()
        
        except CancelBatchException: self.handle("after_cancel_batch")
        finally:  
            self.handle("after_batch")
            #print()


    def all_batches(self, dataloader):
        self.dataloader = dataloader
        self.iters = len(dataloader)
        self.iter = 0
        try:
            self.handle("begin_epoch")
            for xb, yb, *args in self.dataloader:
                self.one_batch(xb, yb)
        except CancelEpochException: self.handle("after_cancel_epoch")
    
     
    def fit(self, epochs,train=True,start_epoch=0):
        self.epochs = epochs 
        self.start_epoch = start_epoch
        try:
            if train:
               self.handle("begin_fit")        
               for epoch in range(start_epoch, epochs):
                   self.epoch = epoch
                   self.learn.model.train()
                   self.in_train = True
                
                   self.all_batches(self.learn.data.train_dl) 
                
                   self.learn.model.eval()
                   self.in_train = False            
                   with torch.no_grad():        
                       self.all_batches(self.learn.data.valid_dl)
       
                   self.handle("after_epoch")
            else:
                self.learn.model.eval()
                self.in_train = False
                with torch.no_grad():
                    self.epoch = 1
                    start = time.time() 
                    self.all_batches(self.learn.data.train_dl)
                    end = time.time()
                    print(end-start)
                    self.handle("after_epoch")

        except CancelTrainException: self.handle("after_cancel_train")
        finally:
            self.handle("after_fit")

    
    def handle(self, name, *args):
        if not self.cbs: return
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, name, None)
            if f is not None: 
                f(*args)
            

