import torch.cuda.nvtx as nvtx

model_list = {
    'small': {'d_model': 768, 'd_ff': 3072, 'n_layers': 12, 'n_heads': 12}, 
    'medium': {'d_model': 1024, 'd_ff': 4096, 'n_layers': 24, 'n_heads': 16}, 
    'large': {'d_model': 1280, 'd_ff': 5120, 'n_layers': 36, 'n_heads': 20}, 
    'xl': {'d_model': 1600, 'd_ff': 6400, 'n_layers': 48, 'n_heads': 25}, 
    '2.7B': {'d_model': 2560, 'd_ff': 10240, 'n_layers': 32, 'n_heads': 32}
}

context_lens = [128, 256, 512, 1024]

def profile_forward(context_len, name, warmup, n):
    model = BasicsTransformerLM(
        10000, context_length, model_list[name]['d_model'], model_list[name]['n_layers'], 
        model_list[name]['n_heads'], model_list[name]['d_ff'], 10000
    )

    model.to(device)
    batch = torch.randint(low=0, high=10000, size=(4, context_length), device=device)
    torch.cuda.synchronize()

    with nvtx.range("warmup"):
        for _ in range(warmup):
            preds = model(batch)
            torch.cuda.synchronize()

    with nvtx.range("forward"):
        for _ in range(n):
            preds = model(batch)
            torch.cuda.synchronize()

if __name__ == '__main__':
    # sweep across the models and context lengths 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--len", type=int)
    args = parser.parse_args()

    profile_forward(args.len, args.model, 15, 30)