import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits, gumbel_noise, temperature, hard=False):
    y = logits + gumbel_noise
    y = F.softmax(y / temperature, dim=-1)
    if not hard:
       return y
    else:
      idx = torch.argmax(y)
      y_hard = torch.zeros_like(y).cuda()
      y_hard.scatter_(0, idx, 1)
      return y_hard

def printModel(path):
    state_dict = torch.load(path)
    gumbel = None
    noise = None
    mask = False
    i = 0
    compression_list = []
    for k,v in state_dict.items():
        #if(len(v.shape)==4):
        #  print(v.shape)
        if(k.find('initial_layers')!=-1):
           continue
        if(len(v.shape)==4 and v.shape[0]>64 and v.shape[1]>64):
          mask = True
        if(k.find("gumbel_weight")!=-1):
          gumbel = v
        if(k.find("gumbel_noise")!=-1):
          index = torch.argmax(gumbel_softmax(gumbel, v, 0.01, True))
          # print(index, 2**(2+index))
          if mask:
             compression_list += [int(2**(2+index))]
    return compression_list


if __name__ == "__main__":
   compression_list = printModel('/scratch/un270/model/Adjoint-Experiments/Nas/1e-7/cifar-4-8-16-32-noschedular-1e-10-sq/50.pt')
   print(compression_list)
