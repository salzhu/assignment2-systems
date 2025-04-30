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

def time_optimizer(model, inputs, targets, w, n):

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    times = []

    for step in range(w+n):

        torch.cuda.synchronize()

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

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model.to(device)

    inputs = torch.randint(0,vocab_size,(batch_size, context_length), device=device)
    targets = torch.randint(0,vocab_size,(batch_size, context_length), device=device)

    time = time_optimizer(model, inputs, targets, 30, 70, sharded=args.sharded)
    print(f'sharded {args.sharded}')
    print(f'time {time * 1000} ms')