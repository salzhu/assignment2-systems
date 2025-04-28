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

        for param in params:
            self.params.append(param)

        for param in self.params:
            print(param)
            if param.device == 'cuda:1':
                self.rank = 1
            if param.requires_grad:
                print('getting here')
                total_params += np.prod(param.data.shape)

        # cur_count = 0 
        # self.index = 0 
        # for param in params:
        #     if param.requires_grad:
        #         cur_count += np.prod(param.data.shape)
        #     if cur_count > total_params / 2: 
        #         break 
        #     self.index += 1

        # self.opt = None 
        # if self.rank == 0: 
        #     self.opt = optimizer_cls(
        #         params[:self.index], **kwargs
        #     )
        # elif self.rank == 1:
        #     self.opt = optimizer_cls(
        #         params[self.index:], **kwargs
        #     )

        self.params_list_0 = []
        self.params_list_1 = []

        cur_count = 0 
        print("total params")
        print(total_params)
        print(params)

        # for param in params:
        #     print(param)

        for param in self.params:
            print('here')
            if param.requires_grad:
                cur_count += np.prod(param.data.shape)
            print('count')
            print(cur_count)
            if cur_count < total_params / 2:
                self.params_list_0.append(param)
            else:
                self.params_list_1.append(param)

        # make two optimizers with the different params 
        self.opt = None 

        if self.rank == 0: 
            self.opt = optimizer_cls(
                self.params_list_0, **kwargs
            )
        elif self.rank == 1:
            self.opt = optimizer_cls(
                self.params_list_1, **kwargs
            )

        print(self.params_list_0)
        print(self.params_list_1)

        # raise NotImplementedError
    
    def step(self, closure, **kwargs):

        # for current rank, they have the params list
        # collect the gradients all_reduce on these params 
        # perform an optimizer step 
        # send these params out 

        self.opt.step()
        # half the parameters are now updated 

        if self.rank == 0: 
            for param in self.params_list_0:
                dist.broadcast(tensor=param.data,src=0)
        elif self.rank == 1: 
            for param in self.params_list_1:
                dist.broadcast(tensor=param.data,src=1)

        # if self.rank == 0: 
        #     for param in self.params[:self.index]:
        #         dist.broadcast(tensor=param.data,src=0)
        # elif self.rank == 1: 
        #     for param in self.params[self.index:]:
        #         dist.broadcast(tensor=param.data,src=1)

        # raise NotImplementedError
    
    def add_param_group(self, param_group: dict[str, Any]):
        raise NotImplementedError