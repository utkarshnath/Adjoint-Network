from datablock import compose
import torch

class Optimizer():
    def __init__(self, params, step_funs, **defaults):
        self.param_groups = list(params)
        if not isinstance(self.param_groups[0], list): self.param_groups = [self.param_groups]
        
        self.step_funs = step_funs
        self.hypers = [{**defaults} for pg in self.param_groups]

    def grad_params(self):
        return [ (p, hyper) for (pg, hyper) in zip(self.param_groups, self.hypers) for p in pg if p.grad is not None]

    def zero_grad(self):
        for (p, hyper) in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        for (p, hyper) in self.grad_params(): compose(p, self.step_funs, **hyper)


def sgd_step(p, lr, **kwargs):
    p.data.add_(-lr, p.grad.data)
    return p

def weight_decay(p, lr, wd, **kwargs):
    p.data.mul_(1-lr*wd)
    return p


class StatefulOptimizer(Optimizer):
    def __init__(self, params, step_funs, stats=None, **defaults):
        super().__init__(params, step_funs, **defaults)
        self.stats = list(stats)
        self.states = {}

    def step(self):
        for (p, hyper) in self.grad_params():
            if p not in self.states:
                state = {}
                for stat in self.stats: state = stat.init_state(p, state)
                self.states[p] = state
 
            state = self.states[p] 
            for stat in self.stats: state = stat.update(p, state, **hyper)
            compose(p, self.step_funs, **state, **hyper)    


class Stat():
    def init_state(self, p, state): raise NotImplementedError
    def update(self, p, state, **kwargs): raise NotImplementedError


class AverageGrad(Stat):
    def __init__(self, name="grad_avg"): self.name = name

    def init_state(self, p, state):
        state[self.name] = torch.zeros_like(p.grad.data)
        return state

    def update(self, p, state, beta1, **kwargs):
        state[self.name].mul_(beta1).add_(1-beta1, p.grad.data)
        return state

class AverageSqGrad(AverageGrad):
    def __init__(self, name="sq_grad_avg"): super().__init__(name)
    def init_state(self, p, state): return super().init_state(p, state)

    def update(self, p, state, beta2, **kwargs):
        state[self.name].mul_(beta2).addcmul_(1-beta2, p.grad.data, p.grad.data)
        return state

class StepCount(Stat):
    def __init__(self, name="step"): self.name = name

    def init_state(self, p, state):
        state[self.name] = 0
        return state

    def update(self, p, state, **kwargs):
        state[self.name] += 1
        return state

def debias(beta, step): return (1 - beta**step)

def adam_step(p, lr, grad_avg, sq_grad_avg, step, eps, beta1, beta2, **kwargs):
    db1 = debias(beta1, step)
    db2 = debias(beta2, step)
    p.data.addcdiv_(-lr / db1, grad_avg, (sq_grad_avg/db2).sqrt() + eps)
    return p
