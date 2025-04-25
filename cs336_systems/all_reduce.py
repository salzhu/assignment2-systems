import os
import numpy as np 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import threading
# from torch_util import get_device

import timeit 
import pandas as pd 

result = []
lock = threading.Lock()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # gloo, nccl

def all_reduce_time(rank, data, world_size):
    
    data.to(f"cuda:{rank}") # moving needs to happen outside. data is setup in another function
        
    dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize()

def cleanup():
    dist.destroy_process_group()

def all_reduce(rank, world_size, tensor, result):
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    # Create tensor
    tensor = torch.randn(tensor.shape).cuda(rank)
    # Warmup
    # tensor.to(f'cuda:{rank}')
    # tensor.to
    for i in range(5):
        dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA kernels to finish
            dist.barrier()            # Wait for all the processes to get here
    
    # Perform all-reduce
    start_time = timeit.default_timer()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = timeit.default_timer()
    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}) took {duration}", flush=True)
    # dist.all_gather_into_tensor(output_tensor=output, input_tensor=torch.tensor(duration), async_op=False)
    all_times = [torch.empty(1,device='cuda') for _ in range(world_size)]
    dist.all_gather(tensor_list=all_times, tensor=torch.tensor([duration]).cuda(rank), async_op=False)
    # result = np.mean(all_times)
    result.append(np.mean(all_times))

    cleanup()

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = {'num. processes': [], 'data size': [], 'time (ms)': []}

    world_sizes = [2, 4, 6]
    shapes = [
        (250, 1000, '1MB'), # 1mb
        (2500, 1000, '10MB'), # 10mb
        (2500, 10000, '100MB'),  # 100mb
        (25000, 10000, '1GB') # 1 gb
    ]

    for world_size in world_sizes: 
        for shape in shapes: 
            manager = mp.Manager()
            result = manager.list()

            data = torch.rand((shape[0],shape[1]), dtype=torch.float32,device=device)

            mp.spawn(all_reduce, args=(world_size,data,result), nprocs=world_size, join=True)
            
            df['num. processes'].append(world_size)
            df['data size'].append(shape[2])
            df['time (ms)'].append(1000 * result[-1])

            print(f'done {world_size} {shape[2]} {1000 * result[-1]}')

    df = pd.DataFrame(df)
    print(df.to_latex(index=False))

    # world_size = 4 
    # result = 0 

    # manager = mp.Manager()
    # result = manager.list()
    # mp.spawn(all_reduce, args=(4,torch.rand(1000),result), nprocs=4, join=True)
    # print(result)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # df = {'num. processes': [], 'data size': [], 'time (ms)': []}

    # world_sizes = [2, 4, 6]
    # shapes = [
    #     (250, 1000, '1MB'), # 1mb
    #     (2500, 1000, '10MB'), # 10mb
    #     (2500, 10000, '100MB'),  # 100mb
    #     (25000, 10000, '1GB') # 1 gb
    # ]

    # for world_size in world_sizes: 
    #     for shape in shapes: 

    #         data = torch.rand((shape[0],shape[1]), dtype=torch.float32,device=device)

    #         start = timeit.default_timer()
    #         mp.spawn(fn=all_reduce_time, args=(data, world_size, ), nprocs=world_size, join=True)
    #         time = timeit.default_timer() - start

    #         mp.spawn(fn=all_reduce_time, args=(time, world_size, ), nprocs=world_size, join=True)
            
    #         df['num. processes'].append(world_size)
    #         df['data size'].append(shape[2])
    #         df['time (ms)'].append(1000 * time)

    #         print(f'done {world_size} {shape[2]} {1000 * time}')

    # df = pd.DataFrame(df)
    # print(df.to_latex(index=False))