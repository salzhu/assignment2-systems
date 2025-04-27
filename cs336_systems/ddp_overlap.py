import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import timeit 
import pandas as pd 
import numpy as np 
import argparse

class DDPIndividualParameters(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        # broadcast weights before training 
        # register a bunch of hooks on each param where the function is an all reduce

        rank = 0 
        # if module.device == 'cuda'

        print('DEVICE DEVICE')
        # print(module.device)

        # if module.device == 'cuda:0':
        #     rank = 1

        self.handles = []
        self.module = module

        # for param in self.module.parameters():
        #     print(param)
        #     if str(param.device) == 'cuda:0':
        #         for dst in range(1, 2):
        #             dist.send(tensor=param, dst=dst)
        #             print(f"Rank {rank} sent data to rank {dst}")
        #     else:
        #         dist.recv(tensor=param, src=0)

        

        # print('------------------------------------------------')
        # print(self.module.parameters())
        # for param in self.module.parameters():
        #     print(param.grad)
        print('------------------------------------------------')

        for param in self.module.parameters():
            print(param.device)
            if param.requires_grad:
                # reduce_handle = dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=True)
                handle = param.register_post_accumulate_grad_hook(lambda p: dist.all_reduce(tensor=p.grad, op=dist.ReduceOp.AVG, async_op=True))
                self.handles.append(handle)

        # raise NotImplementedError
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
        raise NotImplementedError
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()