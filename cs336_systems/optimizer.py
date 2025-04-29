import os
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from typing import Any
from torch.optim import Optimizer as Optimizer

class OptimizerSharded(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs: Any):

        self.rank = dist.get_rank()
        
        self.defaults = dict(**kwargs)

        self.local_param_groups = []
        self.param_count = 0

        hyperparams = dict(**kwargs) 
        super().__init__(params, defaults=kwargs)

        self.params_order = []

        self.opt = optimizer_cls(
            self.local_param_groups, **kwargs
        )
    
    def step(self, closure=None, **kwargs):

        self.opt.step(closure=closure, **kwargs)
        dist.barrier()

        index = 0 
        for param in self.params_order:
            source_rank = index % 2
            dist.broadcast(tensor=param.data,src=source_rank)
            index += 1 

        dist.barrier()

    
    def add_param_group(self, param_group: dict[str, Any]):

        local_param_group = []
        
        for param in param_group['params']:
            if self.param_count % 2 == 0 and self.rank == 0: 
                local_param_group.append(param)
                print('0')
            elif self.param_count % 2 == 1 and self.rank == 1: 
                local_param_group.append(param)
                print('1')
            self.params_order.append(param)
            # index += 1

        param_group_copy = copy.deepcopy(param_group)
        param_group_copy['params'] = copy.deepcopy(local_param_group)
        self.local_param_groups.append(param_group_copy)

        super().add_param_group(param_group)