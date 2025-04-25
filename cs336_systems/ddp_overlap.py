import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import timeit 
import pandas as pd 
import numpy as np 
import argparse

class DDPIndividualParameters:

    def __init__(self, module: torch.nn.Module):
        # broadcast weights before training 
        # register a bunch of hooks on each param where the function is an all reduce
        self.handles = []
        self.module = module

        print('------------------------------------------------')
        print(self.module.parameters())
        for param in self.module.parameters():
            print(param.grad)
        print('------------------------------------------------')

        for param in self.module.parameters():
            reduce_handle = dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=True)
            handle = param.register_post_accumulate_grad_hook(reduce_handle)
            self.handles.append(handle)

        raise NotImplementedError
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
        raise NotImplementedError
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()