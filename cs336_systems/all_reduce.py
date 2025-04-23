import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import timeit 
import pandas as pd 

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # gloo

def all_reduce_time(rank, data, world_size):
    setup(rank, world_size)
    
    # print(f"rank {rank} data (before all-reduce): {data}")
    
    dist.all_reduce(data, async_op=False)
    # print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":

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

            data = torch.rand((shape[0],shape[1]), dtype=torch.float32)

            start = timeit.default_timer()
            mp.spawn(fn=all_reduce_time, args=(data, world_size, ), nprocs=world_size, join=True)
            time = timeit.default_timer() - start
            
            df['num. processes'].append(world_size)
            df['data size'].append(shape[2])
            df['time (ms)'].append(1000 * time)

            print(f'done {world_size} {shape[2]} {1000 * time}')

    df = pd.DataFrame(df)
    print(df.to_latex(index=False))