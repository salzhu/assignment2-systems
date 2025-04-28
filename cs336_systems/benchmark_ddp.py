import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import timeit 
import pandas as pd 
import numpy as np 
import argparse

from cs336_basics.model import BasicsTransformerLM 
from cs336_basics.nn_utils import cross_entropy
from ddp_overlap import DDPIndividualParameters, DDPOverlapBucketed

if __name__ == '__main__':
    # make XL model 
    # load it as DDP 
    # generate data 

    # bunch of passes timed 

    w = 20
    n = 50
    batch_size = 4
    context_length = 256

    vocab_size = 10000
    d_model = 1600
    num_layers = 48
    d_ff = 6400
    num_heads = 25
    rope_theta = 10000

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    ddp_model = DDPIndividualParameters(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.randint(0,vocab_size,(1, batch_size, context_length), device=device)
    targets = torch.randint(0,vocab_size,(1, batch_size, context_length), device=device)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)  # Each rank has own optimizer state
    times = []

    for i in range(w+n):
        
        torch.cuda.synchronize()
        start_time_step = timeit.default_timer()
        outputs = ddp_model(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        end_time = timeit.default_timer()
        times.append((end_time - start_time_step) * 1000)

    print('step time')
    print(np.mean(times))

    