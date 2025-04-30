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

    times = []

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

    cleanup()
    return np.mean(times)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sharded", type=bool, default=False)
    parser.add_argument("--len", type=int, default=256)
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
    times = manager.list()

    world_size = 2

    mp.spawn(time_optimizer_main, args=(world_size,
                                        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                                        times), 
                    nprocs=world_size, join=True)

    print(f'sharded {args.sharded}')
    print(f'time {np.mean(times) * 1000} ms')