import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx

import timeit 
import pandas as pd 
import numpy as np 
import argparse
from torch.profiler import profile

from cs336_basics.model import BasicsTransformerLM 
from cs336_basics.nn_utils import cross_entropy
from ddp_overlap import DDPIndividualParameters, DDPOverlapBucketed

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "28500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # gloo

def cleanup():
    dist.destroy_process_group()

def ddp_overlap_main(rank, world_size, data_in, data_targ, 
                          vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, batch_size, 
                          state_dicts, step_times):
    # torch.cuda.set_device(rank)
    setup(rank, world_size)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    # batch_size = data.size(0)  # @inspect batch_size
    # num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = batch_size // world_size  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data_in = data_in[:,start_index:end_index,:].cuda(rank) # n_steps x 2 x data
    data_targ = data_targ[:,start_index:end_index,:].cuda(rank) # n_steps x 2 x data

    # use params passed in to load a transformer 
    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    # model.load_state_dict(weights)
    model = model.cuda(rank)
    ddp_model = DDPIndividualParameters(model)
    dist.barrier()

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)  # Each rank has own optimizer state

    num_steps = data_in.shape[0]
    warmup_steps = 40


    for step in range(warmup_steps):
        optimizer.zero_grad()
        torch.cuda.synchronize()

        print(f"Rank {rank} train step {step}")
        # Forward pass
        inputs = data_in[step]
        targets = data_targ[step]

        outputs = ddp_model(inputs)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = cross_entropy(outputs, targets)
        loss.backward()
        ddp_model.finish_gradient_synchronization() 
        
        # Update parameters
        optimizer.step()
        
        params = ddp_model.state_dict()
        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, ", flush=True)

    # torch.cuda.memory._record_memory_history(max_entries=1000000)
    dist.barrier()
    
    with nvtx.range("record"):
        for step in range(warmup_steps, num_steps):
            torch.cuda.synchronize()

            start_time_step = timeit.default_timer()

            print(f"Rank {rank} train step {step}")
            # Forward pass
            with nvtx.range(f"forward{step}"):
                inputs = data_in[step]
                targets = data_targ[step]

                outputs = ddp_model(inputs)

                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                
                loss = cross_entropy(outputs, targets)
            with nvtx.range(f"backward{step}"):
                loss.backward()

            with nvtx.range(f"gradsync{step}"):
                ddp_model.finish_gradient_synchronization() 
            
            # Update parameters
            optimizer.step()
            
            end_time_step = timeit.default_timer()
            torch.cuda.synchronize()
            
            params = ddp_model.state_dict()
            print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, ", flush=True)

            step_times.append(end_time_step - start_time_step)

    # torch.cuda.memory._dump_snapshot(f"ddp_{context_length}_{rank}_overlap.pickle")
    # torch.cuda.memory._record_memory_history(enabled=None)
    
    # state_dicts.append(model.state_dict())
    cleanup()

def ddp_overlap_bucketed_main(rank, world_size, data_in, data_targ, 
                          vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, batch_size, 
                          state_dicts, step_times, bucket_size):
    # torch.cuda.set_device(rank)
    setup(rank, world_size)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    # batch_size = data.size(0)  # @inspect batch_size
    # num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = batch_size // world_size  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data_in = data_in[:,start_index:end_index,:].cuda(rank) # n_steps x 2 x data
    data_targ = data_targ[:,start_index:end_index,:].cuda(rank) # n_steps x 2 x data

    # use params passed in to load a transformer 
    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    # model.load_state_dict(weights)
    model = model.cuda(rank)
    ddp_model = DDPOverlapBucketed(model, bucket_size)
    dist.barrier()

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)  # Each rank has own optimizer state

    num_steps = data_in.shape[0]
    warmup_steps = 20

    dist.barrier()


    for step in range(warmup_steps):
        torch.cuda.synchronize()

        print(f"Rank {rank} train step {step}")
        # Forward pass
        inputs = data_in[step]
        targets = data_targ[step]

        outputs = ddp_model(inputs)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = cross_entropy(outputs, targets)
        loss.backward()
        ddp_model.finish_gradient_synchronization() 
        
        # Update parameters
        optimizer.step()
        
        params = ddp_model.state_dict()
        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, ", flush=True)

    # torch.cuda.memory._record_memory_history(max_entries=1000000)
    dist.barrier()
    
    for step in range(warmup_steps, num_steps):
        torch.cuda.synchronize()

        start_time_step = timeit.default_timer()

        print(f"Rank {rank} train step {step}")
        # Forward pass
        inputs = data_in[step]
        targets = data_targ[step]

        outputs = ddp_model(inputs)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = cross_entropy(outputs, targets)
        loss.backward()
        ddp_model.finish_gradient_synchronization() 
        
        # Update parameters
        optimizer.step()
        
        end_time_step = timeit.default_timer()
        torch.cuda.synchronize()
        
        params = ddp_model.state_dict()
        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, ", flush=True)

        step_times.append(end_time_step - start_time_step)

    # torch.cuda.memory._dump_snapshot(f"ddp_{context_length}_{rank}_overlap.pickle")
    # torch.cuda.memory._record_memory_history(enabled=None)
    
    # state_dicts.append(model.state_dict())
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucketed", type=int, default=0)
    parser.add_argument("--bucket_size", type=int, default=1)
    args = parser.parse_args()


    # make XL model 
    # load it as DDP 
    # generate data 

    # bunch of passes timed 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size=2

    w = 40
    n = 60
    batch_size = 4
    context_length = 256

    vocab_size = 10000
    d_model = 1600
    num_layers = 48
    d_ff = 6400
    num_heads = 25
    rope_theta = 10000
    
    n_inputs = torch.randint(0,vocab_size,(w+n, batch_size, context_length), device=device)
    n_targets = torch.randint(0,vocab_size,(w+n, batch_size, context_length), device=device)

    manager = mp.Manager()
    state_dicts = manager.list()

    step_times = manager.list()

    if args.bucketed == 0:

        mp.spawn(ddp_overlap_main, args=(world_size,n_inputs, n_targets, 
                                                vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                                                batch_size,
                                                state_dicts, step_times), 
                    nprocs=world_size, join=True)
        print('not bucketed')
    elif args.bucketed == 1:
        mp.spawn(ddp_overlap_bucketed_main, args=(world_size,n_inputs, n_targets, 
                                                vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                                                batch_size,
                                                state_dicts, step_times, args.bucket_size), 
                    nprocs=world_size, join=True)
        print(f'bucketed {args.bucket_size}')
    
    print('step')
    print(np.mean(step_times))

    