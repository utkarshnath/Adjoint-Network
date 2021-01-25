from functools import partial
import torch
from torch import tensor
import math

def annealer(f):
    def _inner(start, end):
        return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos): 
    return start + pos*(end-start)

@annealer
def sched_cos(start, end, pos): 
    return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

@annealer
def sched_no(start, end, pos):  
    return start

@annealer
def sched_exp(start, end, pos): 
    return start * (end/start) ** pos

@annealer
def sched_dec10(start, end, pos):
    return start * 0.1**(pos//(1/3))

def combine_schedules(parts, sched_fns):
    assert sum(parts) == 1.0
    assert len(parts) == len(sched_fns)
    
    parts = tensor(parts)
    parts = torch.cumsum(parts, 0)

    def _inner(pos):
        idx = (pos <= parts).nonzero().min()
        v = 0.0
        if idx > 0:
            v = parts[idx-1] 
        actual_pos = (pos - v) / (parts[idx] - v)
        return sched_fns[idx](actual_pos)

    return _inner



