import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import timeit 
import pandas as pd 
import numpy as np 
import argparse

from typing import Any
from torch.optim import Optimizer as Optimizer

class OptimizerSharded(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs: Any):
        print(dict(**kwargs))
        # super().__init__(params, dict(**kwargs))

        world_size = 2 
        total_params = 0 

        self.rank = 0 
        self.params = []
        self.defaults = dict(**kwargs)

        for param in params:
            self.params.append(param)

        for param in self.params:
            print(param)
            if param.device == 'cuda:1':
                self.rank = 1
            break 
            # if param.requires_grad:
            #     print('getting here')
            #     total_params += np.prod(param.data.shape)

        self.local_param_groups = []


        # make self.local_params variable
        # call init which populates ^ 
        # make self.optim with self.local_params 

        # in add param group 
        # check parameters, make dict that takes out the 'params' 
        # some params list appends the new param in round robin 
        # self.local_params variable is ^ 
        # also call add param group on wrapper 

        # self.param_groups

        hyperparams = dict(**kwargs) 
        # del(hyperparams['params'])
        super().__init__(self.params, hyperparams )

        print(list(hyperparams.keys()))
        print(self.local_param_groups)
        self.opt = optimizer_cls(
            self.local_param_groups, **kwargs
        )
    
    def step(self, **kwargs):

        # for current rank, they have the params list
        # collect the gradients all_reduce on these params 
        # perform an optimizer step 
        # send these params out 

        self.opt.step(**kwargs)
        # half the parameters are now updated 

        if self.rank == 0: 
            for param in self.local_param_groups:
                dist.broadcast(tensor=param.data,src=0)
        elif self.rank == 1: 
            for param in self.local_param_groups:
                dist.broadcast(tensor=param.data,src=1)

        # if self.rank == 0: 
        #     for param in self.params[:self.index]:
        #         dist.broadcast(tensor=param.data,src=0)
        # elif self.rank == 1: 
        #     for param in self.params[self.index:]:
        #         dist.broadcast(tensor=param.data,src=1)

        # raise NotImplementedError
    
    def add_param_group(self, param_group: dict[str, Any]):
        index = 0
        for param in param_group['params']:
            if index % 2 == 0 and self.rank == 0: 
                self.local_param_groups.append(param)
            elif index % 2 == 1 and self.rank == 1: 
                self.local_param_groups.append(param)
            index += 1

        # self.opt.add_param_group(self.local_param_groups)
        # raise NotImplementedError