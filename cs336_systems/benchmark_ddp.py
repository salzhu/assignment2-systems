import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import timeit 
import pandas as pd 
import numpy as np 
import argparse

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

    