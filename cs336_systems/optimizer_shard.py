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
            self.local_param_groups['params'], **kwargs
        )
        # self.opt. add param group? 

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

        # self.params_list_0 = []
        # self.params_list_1 = []

        # cur_count = 0 
        # print("total params")
        # print(total_params)
        # print(params)

        # # for param in params:
        # #     print(param)

        # for param in self.params:
        #     print('here')
        #     if param.requires_grad:
        #         cur_count += np.prod(param.data.shape)
        #     print('count')
        #     print(cur_count)
        #     if cur_count < total_params / 2:
        #         self.params_list_0.append(param)
        #     else:
        #         self.params_list_1.append(param)

        # # make two optimizers with the different params 
        # self.opt = None 

        # if self.rank == 0: 
        #     self.opt = optimizer_cls(
        #         self.params_list_0, **kwargs
        #     )
        # elif self.rank == 1:
        #     self.opt = optimizer_cls(
        #         self.params_list_1, **kwargs
        #     )

        # print(self.params_list_0)
        # print(self.params_list_1)

        # self.param_groups = self.opt.param_groups
        # self._optimizer_step_pre_hooks = self.opt._optimizer_step_pre_hooks
        # self._optimizer_step_post_hooks = self.opt._optimizer_step_post_hooks

        # raise NotImplementedError
    
    def step(self, **kwargs):

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
        index = 0
        for param in param_group:
            if index % 2 == 0 and self.rank == 0: 
                self.local_param_groups.append(param)
            elif index % 2 == 1 and self.rank == 1: 
                self.local_param_groups.append(param)
            index += 1

        # self.opt.add_param_group(self.local_param_groups)
        # raise NotImplementedError