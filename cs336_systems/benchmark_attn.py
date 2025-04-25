import torch 
import numpy as np 

import timeit
import pandas as pd 
import torch.nn as nn
import argparse

from cs336_basics.model import scaled_dot_product_attention

class Attention(nn.Module):
    def forward(self, Q, K, V):
        return scaled_dot_product_attention(Q, K, V)

compiled_attn = torch.compile(scaled_dot_product_attention)
class AttentionCompiled(nn.Module):
    def forward(self, Q, K, V):
        return compiled_attn(Q, K, V)

def pytorch_attn(batch_size, dim, seq_len, n=100, w=10):
    # make the attention module 
    # make random inputs Q, K, V of size batch_size x seq_len x dim 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rand_Q = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    rand_K = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    rand_V = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

    forward_time = []
    backward_time = []

    forward_memory = []
    backward_memory = []

    attn = Attention()

    for i in range(w + n):
        torch.cuda.synchronize()

        start_time = timeit.default_timer()
        out = attn(rand_Q, rand_K, rand_V)
        # print(out)
        # print(torch.sum(out, [0,1,2]))
        torch.cuda.synchronize()
        mid_time = timeit.default_timer()
        
        mid_mem = torch.cuda.max_memory_allocated()
        
        torch.sum(out, [0,1,2]).backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        end_mem = torch.cuda.max_memory_allocated()

        if i >= w:
            forward_time.append(mid_time - start_time)
            backward_time.append(end_time - mid_time)

            forward_memory.append(mid_mem)
            backward_memory.append(end_mem)

        torch.cuda.reset_peak_memory_stats()

    return np.mean(forward_time), np.mean(backward_time), np.mean(forward_memory)

    # for 100 passes
    # forward, backward, memory before and after backward / forward 

def pytorch_compiled_attn(batch_size, dim, seq_len, n=100, w=10):
    # make the attention module 
    # make random inputs Q, K, V of size batch_size x seq_len x dim 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rand_Q = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    rand_K = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    rand_V = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

    forward_time = []
    backward_time = []

    attn = AttentionCompiled()

    for i in range(w + n):
        torch.cuda.synchronize()

        start_time = timeit.default_timer()
        out = attn(rand_Q, rand_K, rand_V)

        torch.cuda.synchronize()
        mid_time = timeit.default_timer()
                
        torch.sum(out, [0,1,2]).backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()

        if i >= w:
            forward_time.append(mid_time - start_time)
            backward_time.append(end_time - mid_time)

    return np.mean(forward_time), np.mean(backward_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", type=int, default=0)
    args = parser.parse_args()

    df = {'model dim': [], 'seq len': [], 'forward time (ms)': [], 'backward time (ms)': [], 'peak memory': []}

    dims = [1024, 2048, 4096, 8192]
    context_lens = [1024, 2048, 4096, 8192, 16384]

    for dim in dims:
        for context_len in context_lens: 
            print(f'dim {dim} len {context_len}...', end=' ')
            try: 
                if args.compile == 0:
                    ft, bt, fm = pytorch_attn(8, dim, context_len)
                elif args.compile == 1:
                    ft, bt, fm = pytorch_compiled_attn(8, dim, context_len)

                df['model dim'].append(dim)
                df['seq len'].append(context_len)
                df['forward time (ms)'].append(1000 * ft)
                df['backward time (ms)'].append(1000 * bt)
                df['peak memory'].append(fm / (1024*1024))
                print(1000 * ft, 1000 * bt, fm / (1024*1024))
            except: 
                df['model dim'].append(dim)
                df['seq len'].append(context_len)
                df['forward time (ms)'].append('oom')
                df['backward time (ms)'].append('oom')
                df['peak memory'].append('oom')
                print('oom')

    df = pd.DataFrame(df)
    print(df.to_latex(index=False))