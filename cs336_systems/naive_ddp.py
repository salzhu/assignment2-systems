import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import timeit 
import pandas as pd 

from cs336_basics.model import BasicsTransformerLM 
from cs336_basics.nn_utils import cross_entropy

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # gloo

def cleanup():
    dist.destroy_process_group()

def data_parallelism_main(rank, world_size, data_in, data_targ, weights, 
                          vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, batch_size):
    # torch.cuda.set_device(rank)
    setup(rank, world_size)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    # batch_size = data.size(0)  # @inspect batch_size
    # num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = batch_size // world_size  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data_in = data_in[:,start_index:end_index].cuda(rank) # n_steps x 2 x data
    data_targ = data_targ[:,start_index:end_index].cuda(rank) # n_steps x 2 x data

    # use params passed in to load a transformer 
    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model.load_state_dict(weights)
    model = model.cuda(rank)

    # train one step 
    # loss.backward() and get gradients, reduce 
    # optimizer.step()

    # vocab of 10000, batch size of 4 
    
    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    # params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Each rank has own optimizer state

    num_steps = data_in.shape[0]

    for step in range(num_steps):
        torch.cuda.synchronize()
        print(f"Rank {rank} train step {step}")
        # Forward pass
        inputs = data_in[step]
        targets = data_targ[step]

        outputs = model(inputs)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = cross_entropy(outputs, targets)
        loss.backward()

        torch.cuda.synchronize()
        
        # Sync gradients across workers (only difference between standard training and DDP)
        for param in model.parameters():
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        torch.cuda.synchronize()
        
        # Update parameters
        optimizer.step()
        params = model.state_dict()
        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {params['layers.1.ln1.weight']}", flush=True)
    
    cleanup()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = 2

    n = 4
    batch_size = 4
    context_length = 256

    vocab_size = 10000
    d_model = 16 # 1600
    num_layers = 2 # 48
    d_ff = 64 # 6400
    num_heads = 2 # 25
    rope_theta = 10000

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

    data_in = torch.randint(0,vocab_size,(n, batch_size, context_length), device=device)
    data_targ = torch.randint(0,vocab_size,(n, batch_size, context_length), device=device)

    mp.spawn(data_parallelism_main, args=(world_size,data_in, data_targ, model.state_dict(),
                                          vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                                          batch_size), 
             nprocs=world_size, join=True)
    
    print("training og")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(n):
        # Forward pass
        inputs = data_in[step]
        targets = data_targ[step]

        outputs = model(inputs)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        params = model.state_dict()
        print(f"[data_parallelism] Rank og: step = {step}, loss = {loss.item()}, params = {params['layers.1.ln1.weight']}", flush=True)
    
    
    # need to figure out how to pass in model parameters (dict)
    # check that the final weights are same
    # check they match a normal trained model 
    # measure training step time and sharing gradients time