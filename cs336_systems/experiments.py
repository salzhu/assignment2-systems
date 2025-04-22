import torch 
import argparse
import pandas as pd 
from torch.profiler import profile

from benchmarking import run_end_to_end_benchmark, memory_profiling

parser = argparse.ArgumentParser()
parser.add_argument("--part", type=str, default='1.1.3')

# parser.add_argument('--models', 
#                     nargs='+', 
#                     default=['small', 'medium', 'large', 'xl', '2.7B'])
# parser.add_argument('--context_lens', type=int_list,
#                     nargs='+', 
#                     default=[128, 256, 512, 1024])

# time profiling
parser.add_argument("--warmup_its", type=int, default=5)
parser.add_argument("--time_its", type=int, default=10)

# mixed precision 
parser.add_argument("--cast", type=bool, default=False)
parser.add_argument("--dtype", type=str, default='torch.float32')

# memory profiling 
parser.add_argument("--full_step", type=bool, default=True)

args = parser.parse_args()

dtype = torch.float32 
if args.dtype == 'torch.bfloat16':
    dtype = torch.bfloat16
if args.dtype == 'torch.float16':
    dtype = torch.float16

model_list = {
    'small': {'d_model': 768, 'd_ff': 3072, 'n_layers': 12, 'n_heads': 12}, 
    'medium': {'d_model': 1024, 'd_ff': 4096, 'n_layers': 24, 'n_heads': 16}, 
    'large': {'d_model': 1280, 'd_ff': 5120, 'n_layers': 36, 'n_heads': 20}, 
    'xl': {'d_model': 1600, 'd_ff': 6400, 'n_layers': 48, 'n_heads': 25}, 
    '2.7B': {'d_model': 2560, 'd_ff': 10240, 'n_layers': 32, 'n_heads': 32}
}

def part_1_1_3(): 

    """
    command to run: 
    uv run cs336_systems/tables.py --warmup_its 1
    """

    df = {'model': [], 'context len': [], 'forward mean (ms)': [], 'forward std (ms)': [], 'backward mean (ms)': [], 'backward std (ms)': []}

    context_lens = [128, 256, 512, 1024]
    models = ['small', 'medium', 'large', 'xl', '2.7B']
    
    for context_len in context_lens:
        for model in models: 
            try: 
                fm, fs, bm, bs = run_end_to_end_benchmark(4, 10000, context_len, 
                                        model_list[model]['d_model'], 
                                        model_list[model]['n_layers'], 
                                        model_list[model]['n_heads'], 
                                        model_list[model]['d_ff'], 10000, 
                                        args.warmup_its, args.time_its,
                                        cast=args.cast,
                                        dtype=args.dtype)
                df['model'].append(model)
                df['context len'].append(context_len)
                df['forward mean (ms)'].append(1000 * fm)
                df['forward std (ms)'].append(1000 * fs)
                df['backward mean (ms)'].append(1000 * bm)
                df['backward std (ms)'].append(1000 * bs)
            except: 
                df['model'].append(model)
                df['context len'].append(context_len)
                df['forward mean (ms)'].append('oom')
                df['forward std (ms)'].append('oom')
                df['backward mean (ms)'].append('oom')
                df['backward std (ms)'].append('oom')

    df = pd.DataFrame(df)
    print(df.to_latex(index=False))

def part_1_1_5(): 

    """
    command to run: 
    uv run cs336_systems/tables.py --part 1.1.5 --cast_115 True --dtype_115 torch.bfloat16
    """

    df = {'model': [], 'context len': [], 'forward mean (ms)': [], 'forward std (ms)': [], 'backward mean (ms)': [], 'backward std (ms)': []}

    context_lens = [256]
    models = ['small', 'medium', 'large', 'xl', '2.7B']

    for context_len in context_lens:
        for model in models: 
            # try: 
            fm, fs, bm, bs = run_end_to_end_benchmark(4, 10000, context_len, 
                                    model_list[model]['d_model'], 
                                    model_list[model]['n_layers'], 
                                    model_list[model]['n_heads'], 
                                    model_list[model]['d_ff'], 10000, 
                                    args.warmup_its, args.time_its,
                                    cast=args.cast,
                                    dtype=dtype)
            df['model'].append(model)
            df['context len'].append(context_len)
            df['forward mean (ms)'].append(1000 * fm)
            df['forward std (ms)'].append(1000 * fs)
            df['backward mean (ms)'].append(1000 * bm)
            df['backward std (ms)'].append(1000 * bs)
            # except: 
            #     df['model'].append(model)
            #     df['context len'].append(context_len)
            #     df['forward mean (ms)'].append('oom')
            #     df['forward std (ms)'].append('oom')
            #     df['backward mean (ms)'].append('oom')
            #     df['backward std (ms)'].append('oom')

    df = pd.DataFrame(df)
    print(df.to_latex(index=False))

def part_1_1_6(): 

    """
    command to run: 
    uv run cs336_systems/tables.py --part 1.1.6 --cast True --dtype torch.bfloat16
    """

    context_lens = [128, 256, 512]
    models = ['2.7B']

    for context_len in context_lens:
        for model in models:
            print(f'running model {model}_{context_len}')
            memory_profiling(f'{model}_{context_len}_forward_mixed',
                             4, 10000, context_len, 
                             model_list[model]['d_model'], 
                             model_list[model]['n_layers'], 
                             model_list[model]['n_heads'], 
                             model_list[model]['d_ff'], 10000, 
                             args.warmup_its, args.time_its, 
                             args.full_step, cast=args.cast, dtype=dtype)


if __name__ == '__main__': 
    if args.part == '1.1.3':
        part_1_1_3()
    elif args.part == '1.1.5':
        part_1_1_5()
    elif args.part == '1.1.6':
        part_1_1_6()
