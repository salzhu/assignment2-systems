import torch 
import numpy as np 

import timeit
import pandas as pd 
import torch.nn as nn
import argparse

from flash_attention import FlashAttentionTriton

def flash_attn_triton(dim, seq_len, dtype, n=100, w=10):
    # make the attention module 
    # make random inputs Q, K, V of size batch_size x seq_len x dim 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rand_Q = torch.randn(1, seq_len, dim, dtype=dtype, device=device, requires_grad=True)
    rand_K = torch.randn(1, seq_len, dim, dtype=dtype, device=device, requires_grad=True)
    rand_V = torch.randn(1, seq_len, dim, dtype=dtype, device=device, requires_grad=True)
    rand_dO = torch.randn(1, seq_len, dim, dtype=dtype, device=device, requires_grad=True)

    forward_time = []
    backward_time = []
    full_time = []

    attn = FlashAttentionTriton

    for i in range(w + n):
        torch.cuda.synchronize()

        start_time = timeit.default_timer()
        out = attn.apply(rand_Q, rand_K, rand_V, True)
        # print(out)
        # print(torch.sum(out, [0,1,2]))
        torch.cuda.synchronize()
        mid_time = timeit.default_timer()

        out.backward(rand_dO)
                
        # torch.sum(out, [0,1,2]).backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()

        if i >= w:
            forward_time.append(mid_time - start_time)
            backward_time.append(end_time - mid_time)
            full_time.append(end_time - start_time)

    return np.mean(forward_time), np.mean(backward_time), np.mean(full_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--triton", type=int, default=1)
    args = parser.parse_args()

    df = {'model dim': [], 'seq len': [], 'dtype': [], 'forward time (ms)': [], 'backward time (ms)': [], 'full step time (ms)': []}

    dims = [
        16,
        32,
        64,
        128
    ]
    context_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    # context_lens = [128, 256, 1024, 4096, 16384, 65536]
    dtypes = [torch.float32, torch.bfloat16]
    dtypes = [torch.bfloat16]

    for dtype in dtypes: 
        for dim in dims:
            for context_len in context_lens: 
                print(f'dim {dim} len {context_len}...', end=' ')
                # try: 
                # if args.triton == 1:
                #     ft, bt, fm = pytorch_attn(8, dim, context_len)
                # elif args.compile == 1:
                #     ft, bt = pytorch_compiled_attn(8, dim, context_len)

                # use tile size of 16 for all
                with torch.autocast(device_type='cuda',dtype=dtype)
                    forward, backward, full = flash_attn_triton(dim, context_len, dtype)

                    df['model dim'].append(dim)
                    df['seq len'].append(context_len)
                    df['dtype'].append(str(dtype))
                    df['forward time (ms)'].append(1000 * forward)
                    df['backward time (ms)'].append(1000 * backward)
                    df['full step time (ms)'].append(1000 * full)

                    print(1000 * forward, 1000 * backward)
                    # except: 
                    #     df['model dim'].append(dim)
                    #     df['seq len'].append(context_len)
                    #     df['forward time (ms)'].append('oom')
                    #     df['backward time (ms)'].append('oom')
                    #     df['peak memory'].append('oom')
                    #     print('oom')

        df = pd.DataFrame(df)
        print(df.to_latex(index=False))