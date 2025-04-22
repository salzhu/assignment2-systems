import argparse
import torch
import wandb
import os
import numpy as np 
from tqdm import tqdm
import timeit
from contextlib import nullcontext
from torch.profiler import profile

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

def end_to_end_benchmark(model, data, w, n, cast=False, dtype=torch.float32):

    forward_pass_time = []
    backward_pass_time = []

    cm = torch.autocast(device_type='cuda',dtype=dtype) if cast else nullcontext()

    with cm:

        torch.cuda.synchronize()

        # all run on the same batch of (pre-generated) data

        # w warmup steps, no timing
        for i in range(w):
            preds = model(data)
            torch.cuda.synchronize()
            preds.mean().backward()
            torch.cuda.synchronize()

        # n steps, timed 
        for i in range(n):

            start = timeit.default_timer()
            # forward pass of model on data 
            preds = model(data)
            torch.cuda.synchronize()
            end = timeit.default_timer()

            # backward pass on predictions
            preds.mean().backward()
            backward_pass_time.append(timeit.default_timer() - end)
            torch.cuda.synchronize()
            forward_pass_time.append(end - start)

            torch.cuda.synchronize()

    return forward_pass_time, backward_pass_time

def run_end_to_end_benchmark(batch_size, vocab_size, context_length, d_model, n_layers, n_heads, d_ff, rope_theta, warmup_its, time_its, cast=False, dtype=torch.float32):
    # initialize transformer model
    model = BasicsTransformerLM(
        vocab_size, context_length, d_model, n_layers, n_heads, d_ff, rope_theta
    )

    # generate random batch of data
    # default: batch size of 4, vocab size of 10000
    batch = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch.to(device)

    forward_pass_time, backward_pass_time = end_to_end_benchmark(model, batch, warmup_its, time_its, cast=cast, dtype=dtype)

    print(f'forward pass timing stats | mean {np.mean(forward_pass_time)}s | std {np.std(forward_pass_time)}s')
    print(f'backward pass timing stats | mean {np.mean(backward_pass_time)}s | std {np.std(backward_pass_time)}s')

    return np.mean(forward_pass_time), np.std(forward_pass_time), np.mean(backward_pass_time), np.std(backward_pass_time)

def memory_profiling(name, batch_size, vocab_size, context_length, d_model, n_layers, n_heads, d_ff, rope_theta, warmup_its, n_steps, full_step=True, cast=False, dtype=torch.float32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Start recording memory history.
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # initialize model, data
    model = BasicsTransformerLM(
        vocab_size, context_length, d_model, n_layers, n_heads, d_ff, rope_theta
    )
    model.to(device)
    batch = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
    # optimizer = AdamW(
    #     model.parameters()
    # )
    torch.cuda.synchronize()

    # warm-up phase in your benchmarking script

    cm = torch.autocast(device_type='cuda',dtype=dtype) if cast else nullcontext()

    with cm:

        torch.cuda.synchronize()

        # all run on the same batch of (pre-generated) data

        # w warmup steps, no timing
        for i in range(warmup_its):
            # optimizer.zero_grad()
            preds = model(batch)
            torch.cuda.synchronize()
            # preds.mean().backward()
            # optimizer.step()
            # torch.cuda.synchronize()

        # n steps, timed 

        # Start recording memory history.
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        for i in range(n_steps):

            # forward pass of model on data 
            # optimizer.zero_grad()
            preds = model(batch)
            torch.cuda.synchronize()

            # backward pass on predictions
            # preds.mean().backward()
            # optimizer.step()
            # torch.cuda.synchronize()

    # Save a pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot(f"{name}.pickle")

    # Stop recording history.
    torch.cuda.memory._record_memory_history(enabled=None)

def memory_profiling2(name, batch_size, vocab_size, context_length, d_model, n_layers, n_heads, d_ff, rope_theta, warmup_its, n_steps, full_step=True, cast=False, dtype=torch.float32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Start recording memory history.
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    # n_steps = 3

    # initialize model, data
    model = BasicsTransformerLM(
        vocab_size, context_length, d_model, n_layers, n_heads, d_ff, rope_theta
    )
    model.to(device)
    batch = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
    torch.cuda.synchronize()
    
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=n_steps),
        schedule=torch.profiler.schedule(wait=0, warmup=warmup_its, active=n_steps, repeat=n_steps),
        # experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # cm = torch.autocast(device_type='cuda',dtype=dtype) if cast else nullcontext()
        # with cm:
        torch.cuda.synchronize()
        for i in range(n_steps):
            # try:
            print(name, n_steps, i)
            # run model on a batch of data...
            torch.cuda.synchronize()
            
            preds = model(batch)
            del preds
            torch.cuda.synchronize()
            
            # if full_step:
            #     preds.mean().backward()
            #     torch.cuda.synchronize()

            prof.step()
            torch.cuda.synchronize()
                # prof.export_memory_timeline(f"{name}_timeline.html", device=device)
            # except Exception as e:
            #     print(f"blah An error occurred: {e}")
        
        # Save a graphical timeline of memory usage.
        print('got here')
        prof.export_memory_timeline(f"{name}_timeline.html", device=device)
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     print(f"Error occurred during iteration {i}")
            
    # Save a pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot(f"{name}_memory_snapshot.pickle")
    # Stop recording history.
    torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=int, default=10000)

    # benchmarking parameters 
    parser.add_argument("--warmup_its", type=int, default=5)
    parser.add_argument("--time_its", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cast", type=bool, default=False)
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float32)

    args = parser.parse_args()

    run_end_to_end_benchmark(
        args.batch_size, 
        args.vocab_size, args.context_length, args.d_model, args.n_layers, args.n_heads, args.d_ff, args.rope_theta, 
        args.warmup_its, args.time_its, cast=args.cast, dtype=args.dtype
    )