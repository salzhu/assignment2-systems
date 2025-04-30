import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx

import timeit 
import pandas as pd 
import numpy as np 
import argparse

from cs336_basics.model import BasicsTransformerLM 
from cs336_basics.nn_utils import cross_entropy

class DDPIndividualParameters(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        # broadcast weights before training 
        # register a bunch of hooks on each param where the function is an all reduce

        rank = 0 
        # if module.device == 'cuda'

        # print(module.device)

        # if module.device == 'cuda:0':
        #     rank = 1

        self.handles = []
        self.module = module

        for param in self.module.parameters():
            dist.broadcast(tensor=param.data,src=0)
            # if str(param.device) == 'cuda:0':
            #     dist.send(tensor=param.data, dst=1)
            #     print(f"Rank 0 sent data to rank 1")
            # else:
            #     dist.recv(tensor=param.data, src=0)

        
        # print('------------------------------------------------')
        # print(self.module.parameters())
        # for param in self.module.parameters():
        #     print(param.grad)
        # print('------------------------------------------------')

        for param in self.module.parameters():
            if param.requires_grad:
                # handle = dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=True)
                # handle = param.register_post_accumulate_grad_hook(lambda p: dist.all_reduce(tensor=p.grad, op=dist.ReduceOp.AVG, async_op=True))
                # param.register_post_accumulate_grad_hook(lambda p: dist.all_reduce(tensor=p.grad, op=dist.ReduceOp.AVG, async_op=True))
                # handle = param.register_post_accumulate_grad_hook(hook)
                # self.handles.append(handle)
                param.register_post_accumulate_grad_hook(self.add_hook())

        # raise NotImplementedError
    
    def add_hook(self):
        def hook(param):
            with nvtx.range(f"all_reduce"):
                handle = dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)
            # param.grad /= 2
            # handle.wait()
            return None 
        return hook 
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
        raise NotImplementedError
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.div_(2)

        self.handles.clear()

class DDPOverlapBucketed(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, bucket_size):
        super().__init__()

        self.handles = []
        self.module = module
        self.bucket_size = bucket_size

        self.n_floats = bucket_size * 1024 * 1024 / 4

        for param in self.module.parameters():
            dist.broadcast(tensor=param.data,src=0)

        # self.param_counts = []
        # param_list = list(self.module.parameters())
        # for i in range(len(param_list)):
        #     if param_list[i].requires_grad:


        # send at most bucket MB of gradients at a time 
        self.param_buckets = []

        cur_count = 0 
        cur_list = []
        index = 0 

        # for i in range(len(list(self.module.parameters())) - 1):
        #     param = list(self.module.parameters())[i]
        #     if param.requires_grad:
        #         cur_count += np.prod(param.data.shape)
        #         cur_list.append(index)
            
        #     next_param = list(self.module.parameters())[i + 1]
        #     if np.prod(next_param.data.shape) + cur_count > self.n_floats: 
        #         # attach hooks 
        #         self.param_buckets.append(cur_list)
        #         cur_list = []
        #         previous_param.register_post_accumulate_grad_hook(self.add_hook(len(self.param_buckets) - 1))
        #         previous_param = param
            
        #     if cur_count + np.prod(param.data.shape)

        # cur_count = 0 
        # cur_list = []
        # index = 0 
        # last_param = None

        # for param in self.module.parameters():
        #     if param.requires_grad:
        #         last_param = param
        #         cur_count += np.prod(param.data.shape)
        #         cur_list.append(index)

        #         if cur_count >= self.n_floats:
        #             self.param_buckets.append(cur_list)
        #             cur_list = []

        #             # add a hook 
        #             param.register_post_accumulate_grad_hook(self.add_hook(len(self.param_buckets) - 1))
        #             cur_count = 0 

        #         index += 1

        # self.param_buckets.append(cur_list)
        # last_param.register_post_accumulate_grad_hook(self.add_hook(len(self.param_buckets) - 1))


        cur_count = 0 
        cur_list = [0]
        index = 0 
        previous_param = None 
        for param in self.module.parameters():
            if param.requires_grad:
                previous_param = param
                break 

        for param in self.module.parameters():
            if param.requires_grad:

                if cur_count + np.prod(param.data.shape) >= self.n_floats:
                    self.param_buckets.append(cur_list)
                    cur_list = []

                    # add a hook 
                    previous_param.register_post_accumulate_grad_hook(self.add_hook(len(self.param_buckets) - 1))
                    previous_param = param
                    cur_count = 0 

                cur_count += np.prod(param.data.shape)

                if index != 0:
                    cur_list.append(index)
                index += 1

        self.param_buckets.append(cur_list)
        previous_param.register_post_accumulate_grad_hook(self.add_hook(len(self.param_buckets) - 1))

        # pre-compute the buckets 
        # only add hooks to the appropriate ones 

        # for param in self.module.parameters():
        #     if param.requires_grad:
        #         param.register_post_accumulate_grad_hook(self.add_hook())

    def add_hook(self, id):
        def hook(param_hooked):
            # flatten
            param_ids = self.param_buckets[id]
            if param_ids == []:
                return None 
            flat_list = []
            index = 0 
            # print(param_ids)
            for param in self.module.parameters():
                if param.requires_grad:
                    if index in param_ids:
                        flat_list.append(param.grad)
                    index += 1

            # print(flat_list)
            if flat_list != [] and flat_list is not None:
                flat_grads = torch._utils._flatten_dense_tensors(flat_list)
                # all reduce
                handle = dist.all_reduce(tensor=flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                # unflatten 
                unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, flat_list)

                count_grad = 0 
                index = 0 
                for param in self.module.parameters():
                    if param.requires_grad:
                        if index in param_ids:
                            param.grad = unflat_grads[count_grad] # divide 
                            # param.grad.div_(2)
                            count_grad += 1
                        index += 1
            
                self.handles.append(handle)
            return None 
        return hook 
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.div_(2)

        self.handles.clear()

# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--flat", type=int, default=0)
#     # args = parser.parse_args()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     world_size = 2

#     n = 20
#     batch_size = 4
#     context_length = 256

#     vocab_size = 10000
#     d_model = 1600
#     num_layers = 48
#     d_ff = 6400
#     num_heads = 25
#     rope_theta = 10000

#     model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
#     model_ddp = DDPIndividualParameters(model)

#     w = 10
#     n = 20

#     inputs = torch.randint(0,vocab_size,(batch_size, context_length), device=device)
#     targets = torch.randint(0,vocab_size,(batch_size, context_length), device=device)

#     times = []

#     for i in range(w + n):
#         start_time = timeit.default_timer()
#         outputs = model_ddp(inputs)

#         outputs = outputs.view(-1, outputs.size(-1))
#         targets = targets.view(-1)
        
#         loss = cross_entropy(outputs, targets)
#         loss.backward()
#         torch.cuda.synchronize()

#         if i >= w:
#             times.append(timeit.default_timer() - start_time)


#     manager = mp.Manager()
#     state_dicts = manager.list()

#     step_times = manager.list()
#     grad_collect_times = manager.list()

#     mp.spawn(ddp_naive_main, args=(world_size,data_in, data_targ, model.state_dict(),
#                                     vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
#                                     batch_size,
#                                     state_dicts, step_times, grad_collect_times), 
#         nprocs=world_size, join=True)

#     # print(args.flat)

#     # if args.flat == 0:
#     #     print('naive ddp NOT FLAT')
#     #     mp.spawn(ddp_naive_main, args=(world_size,data_in, data_targ, model.state_dict(),
#     #                                         vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
#     #                                         batch_size,
#     #                                         state_dicts, step_times, grad_collect_times), 
#     #             nprocs=world_size, join=True)
#     # elif args.flat == 1:
#     #     print('naive ddp flat')
#     #     mp.spawn(ddp_flat_main, args=(world_size,data_in, data_targ, model.state_dict(),
#     #                                         vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
#     #                                         batch_size,
#     #                                         state_dicts, step_times, grad_collect_times), 
#     #             nprocs=world_size, join=True)
    
#     print('step')
#     print(np.mean(step_times))

#     print('grad collect')
#     print(np.mean(grad_collect_times))
#     #