import os
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import timeit 
import pandas as pd 
import numpy as np 
import argparse

from typing import Any

from optimizer_shard import OptimizerSharded
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import BasicsTransformerLM 

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "28500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # gloo

def cleanup():
    dist.destroy_process_group()

def time_optimizer_main(rank, world_size, 
                        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                        times):

    setup(rank, world_size)

    w = 20 
    n = 40 

    inputs = torch.randint(0,vocab_size,(4, context_length)).cuda(rank)
    targets = torch.randint(0,vocab_size,(4, context_length)).cuda(rank)

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model.cuda(rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(w+n):
        print(step, end= ' ', flush=True)

        torch.cuda.synchronize()
        optimizer.zero_grad()

        start_time = timeit.default_timer()

        outputs = model(inputs)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = cross_entropy(outputs, targets)
        loss.backward()

        optimizer.step()
        torch.cuda.synchronize()

        end_time = timeit.default_timer()

        if step >= w:
            times.append(end_time - start_time)
            print(1000 * (end_time - start_time), end=' ')

    cleanup()
    # return np.mean(times)

def time_optimizer_sharded_main(rank, world_size, 
                        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                        times):

    setup(rank, world_size)

    w = 20 
    n = 40 

    inputs = torch.randint(0,vocab_size,(4, context_length)).cuda(rank)
    targets = torch.randint(0,vocab_size,(4, context_length)).cuda(rank)

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model.cuda(rank)

    optimizer = OptimizerSharded(
        model.parameters(),
        torch.optim.AdamW,
        lr=1e-3
    )

    for step in range(w+n):
        print(step, end= ' ', flush=True)

        torch.cuda.synchronize()
        optimizer.zero_grad()

        start_time = timeit.default_timer()

        outputs = model(inputs)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = cross_entropy(outputs, targets)
        loss.backward()

        optimizer.step()
        torch.cuda.synchronize()

        end_time = timeit.default_timer()

        if step >= w:
            times.append(end_time - start_time)
            print(1000 * (end_time - start_time), end=' ')

    cleanup()

def memory_optimizer_main(rank, world_size, 
                        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                        mems_after_model_initialization, mems_before_optimizer_step, mems_after_optimizer_step):

    setup(rank, world_size)

    w = 20 
    n = 40 

    inputs = torch.randint(0,vocab_size,(4, context_length)).cuda(rank)
    targets = torch.randint(0,vocab_size,(4, context_length)).cuda(rank)

    torch.cuda.reset_peak_memory_stats()

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model.cuda(rank)
    mem_after_model_initialization = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.cuda.reset_peak_memory_stats()

    mems_after_model_initialization.append(mem_after_model_initialization)

    for step in range(w+n):

        torch.cuda.reset_peak_memory_stats()
        
        print(step, end= ' ', flush=True)

        torch.cuda.synchronize()
        optimizer.zero_grad()

        # start_time = timeit.default_timer()

        outputs = model(inputs)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = cross_entropy(outputs, targets)
        loss.backward()

        mem_before_optimizer_step = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        optimizer.step()
        torch.cuda.synchronize()

        mem_after_optimizer_step = torch.cuda.max_memory_allocated()

        # end_time = timeit.default_timer()

        if step >= w:
            mems_before_optimizer_step.append(mem_before_optimizer_step)
            mems_after_optimizer_step.append(mem_after_optimizer_step)
            print(mem_after_optimizer_step)
            # times.append(end_time - start_time)
            # print(1000 * (end_time - start_time), end=' ')

    cleanup()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sharded", type=bool, default=False)
    parser.add_argument("--len", type=int, default=256)
    parser.add_argument("--exp", type=str, default='time')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 4
    context_length = args.len # 256

    vocab_size = 10000
    d_model = 1600
    num_layers = 48
    d_ff = 6400
    num_heads = 25
    rope_theta = 10000

    manager = mp.Manager()

    world_size = 2

    if args.sharded == False and args.exp == 'time':
        times = manager.list()
        mp.spawn(time_optimizer_main, args=(world_size,
                                            vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                                            times), 
                        nprocs=world_size, join=True)
        print(f'sharded {args.sharded}')
        print(f'time {np.mean(times) * 1000} ms')
    elif args.sharded == True and args.exp == 'time':
        times = manager.list()
        mp.spawn(time_optimizer_sharded_main, args=(world_size,
                                            vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                                            times), 
                        nprocs=world_size, join=True)
        print(f'sharded {args.sharded}')
        print(f'time {np.mean(times) * 1000} ms')
    elif args.sharded == False and args.exp == 'memory': 
        mems_after_model_initialization = manager.list()
        mems_before_optimizer_step = manager.list()
        mems_after_optimizer_step = manager.list()
        mp.spawn(memory_optimizer_main, args=(world_size,
                                            vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                                            mems_after_model_initialization, mems_before_optimizer_step, mems_after_optimizer_step), 
                        nprocs=world_size, join=True)
        print(f'sharded {args.sharded}')
        print(f'mem after model init {np.mean(mems_after_model_initialization) * 1024} MB')
        print(f'mem before optimizer step {np.mean(mems_before_optimizer_step) * 1024} MB')
        print(f'mem after optimizer step {np.mean(mems_after_optimizer_step) * 1024} MB')

    