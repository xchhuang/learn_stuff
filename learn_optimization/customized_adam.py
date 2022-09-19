import torch
from torch.optim import Optimizer
import math


class ADAMOptimizer(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999):
        self.parameters = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.EMA1 = [torch.zeros_like(param) for param in self.parameters]
        self.EMA2 = [torch.zeros_like(param) for param in self.parameters]
        self.iter_num = 0
        self.eps = 1e-9

    def step(self):
        self.iter_num += 1

        correct1 = 1 - self.beta1 ** self.iter_num  # EMA1 bias correction.
        correct2 = 1 - self.beta2 ** self.iter_num  # EMA2 bias correction.

        with torch.no_grad():
            for param, EMA1, EMA2 in zip(self.parameters, self.EMA1, self.EMA2):
                EMA1.set_((1 - self.beta1) * param.grad + self.beta1 * EMA1)
                EMA2.set_((1 - self.beta2) * (param.grad ** 2) + self.beta2 * EMA2)

                numenator = EMA1 / correct1
                denominator = (EMA2 / correct2).sqrt() + self.eps

                param -= self.lr * numenator / denominator


class ADAMOptimizer2(Optimizer):
    """
    implements ADAM Algorithm, as a preceding step.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ADAMOptimizer2, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None
        for group in self.param_groups:
            #print(group.keys())
            #print (self.param_groups[0]['params'][0].size()), First param (W) size: torch.Size([10, 784])
            #print (self.param_groups[0]['params'][1].size()), Second param(b) size: torch.Size([10])
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Momentum (Exponential MA of gradients)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    #print(p.data.size())
                    # RMS Prop componenet. (Exponential MA of squared gradients). Denominator.
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                b1, b2 = group['betas']
                state['step'] += 1

                # L2 penalty. Gotta add to Gradient as well.
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Momentum
                exp_avg = torch.mul(exp_avg, b1) + (1 - b1)*grad
                # RMS
                exp_avg_sq = torch.mul(exp_avg_sq, b2) + (1-b2)*(grad*grad)

                denom = exp_avg_sq.sqrt() + group['eps']

                bias_correction1 = 1 / (1 - b1 ** state['step'])
                bias_correction2 = 1 / (1 - b2 ** state['step'])

                adapted_learning_rate = group['lr'] * bias_correction1 / math.sqrt(bias_correction2)

                p.data = p.data - adapted_learning_rate * exp_avg / denom

                # if state['step']  % 10000 ==0:
                #     print ("group:", group)
                #     print("p: ",p)
                #     print("p.data: ", p.data) # W = p.data

        return loss


