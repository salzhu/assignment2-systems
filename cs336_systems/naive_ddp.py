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

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # gloo

def cleanup():
    dist.destroy_process_group()

def ddp_naive_main(rank, world_size, data_in, data_targ, weights, 
                          vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, batch_size, 
                          state_dicts, step_times, grad_collect_times):
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
    warmup_steps = 20

    for step in range(num_steps):
        torch.cuda.synchronize()

        start_time_step = timeit.default_timer()

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

        start_time_grad = timeit.default_timer()
        
        # Sync gradients across workers (only difference between standard training and DDP)
        for param in model.parameters():
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        torch.cuda.synchronize()

        end_time_grad = timeit.default_timer()
        
        # Update parameters
        optimizer.step()
        end_time_step = timeit.default_timer()
        
        params = model.state_dict()
        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {params['layers.1.ln1.weight']}", flush=True)

        if step >= warmup_steps: 
            step_times.append(end_time_step - start_time_step)
            grad_collect_times.append(end_time_grad - start_time_grad)
    
    # state_dicts.append(model.state_dict())
    cleanup()

def ddp_flat_main(rank, world_size, data_in, data_targ, weights, 
                          vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, batch_size, 
                          state_dicts, step_times, grad_collect_times):
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
    warmup_steps = 20

    for step in range(num_steps):
        torch.cuda.synchronize()

        start_time_step = timeit.default_timer()

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

        start_time_grad = timeit.default_timer()
        
        # Sync gradients across workers (only difference between standard training and DDP)
        # flatten first 
        # get flat gradient: param_flat
        # dist.all_reduce(tensor=param_flat, op=dist.ReduceOp.AVG, async_op=False)
        # put back into gradients 

        param_list = model.state_dict()
        
        flat_grads = torch._utils._flatten_dense_tensors([param.grad for param in model.parameters()])
        dist.all_reduce(tensor=flat_grads, op=dist.ReduceOp.AVG, async_op=False)

        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, [tensor for tensor in param_list.values()])
        # print(unflattened[:4])
        for param, tensor in zip(model.parameters(), unflat_grads):
            # new_state_dict[key] = tensor
            param.grad = tensor

        # for param in model.parameters():
        #     dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
        # model.load_state_dict(new_state_dict)

        torch.cuda.synchronize()

        end_time_grad = timeit.default_timer()
        
        # Update parameters
        optimizer.step()
        torch.cuda.synchronize(device=device)
        end_time_step = timeit.default_timer()
        
        params = model.state_dict()
        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {params['layers.1.ln1.weight']}", flush=True)

        if step >= warmup_steps: 
            step_times.append(end_time_step - start_time_step)
            grad_collect_times.append(end_time_grad - start_time_grad)
    
    # state_dicts.append(model.state_dict())
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", type=int, default=0)
    parser.add_argument("--context_length", type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = 2

    n = 80
    batch_size = 4
    context_length = args.context_length # 256

    vocab_size = 10000
    d_model = 1600
    num_layers = 48
    d_ff = 6400
    num_heads = 25
    rope_theta = 10000

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

    data_in = torch.randint(0,vocab_size,(n, batch_size, context_length), device=device)
    data_targ = torch.randint(0,vocab_size,(n, batch_size, context_length), device=device)

    manager = mp.Manager()
    state_dicts = manager.list()

    step_times = manager.list()
    grad_collect_times = manager.list()

    print(args.flat)

    if args.flat == 0:
        print('naive ddp NOT FLAT')
        mp.spawn(ddp_naive_main, args=(world_size,data_in, data_targ, model.state_dict(),
                                            vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                                            batch_size,
                                            state_dicts, step_times, grad_collect_times), 
                nprocs=world_size, join=True)
    elif args.flat == 1:
        print('naive ddp flat')
        mp.spawn(ddp_flat_main, args=(world_size,data_in, data_targ, model.state_dict(),
                                            vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                                            batch_size,
                                            state_dicts, step_times, grad_collect_times), 
                nprocs=world_size, join=True)
    
    print('step')
    print(np.mean(step_times))

    print('grad collect')
    print(np.mean(grad_collect_times))
    
    # print("training og --- check results match!")

    # model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # for step in range(n):
    #     # Forward pass
    #     inputs = data_in[step]
    #     targets = data_targ[step]

    #     outputs = model(inputs)

    #     outputs = outputs.view(-1, outputs.size(-1))
    #     targets = targets.view(-1)
        
    #     loss = cross_entropy(outputs, targets)
    #     loss.backward()
    #     optimizer.step()

    #     params = model.state_dict()
    #     print(f"[data_parallelism] Rank og: step = {step}, loss = {loss.item()}, params = {params['layers.1.ln1.weight']}", flush=True)
    
    # print("done")
    # print("checking params are the same") 
    # model2 = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    # model2.load_state_dict(state_dicts[-1])
    # model2_params = model2.state_dict()

    # for name, param in model.state_dict().items():
    #     print(f"Layer: {name}, match: {param == model2_params[name]}")
    
    # need to figure out how to pass in model parameters (dict)
    # check that the final weights are same
    # check they match a normal trained model 
    # measure training step time and sharing gradients time