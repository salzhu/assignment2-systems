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

        # if rank == 0:
        #     for dst in range(1, size):
        #         dist.send(tensor=tensor + dst, dst=dst)
        #         print(f"Rank {rank} sent data {tensor[0] + dst} to rank {dst}")
        # else:
        #     dist.recv(tensor=tensor, src=0)

        self.handles = []
        self.module = module

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