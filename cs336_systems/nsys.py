import torch 
import torch.cuda.nvtx as nvtx
import argparse
import math 

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from einops import rearrange, einsum

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
        10000, context_len, model_list[name]['d_model'], model_list[name]['n_layers'], 
        model_list[name]['n_heads'], model_list[name]['d_ff'], 10000
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch = torch.randint(low=0, high=10000, size=(4, context_len), device=device)
    torch.cuda.synchronize()

    with nvtx.range("warmup"):
        for i in range(warmup):
            preds = model(batch)
            torch.cuda.synchronize()

    with nvtx.range("forward"):
        for i in range(n):
            preds = model(batch)
            torch.cuda.synchronize()



def profile_backward(context_len, name, warmup, n):
    model = BasicsTransformerLM(
        10000, context_len, model_list[name]['d_model'], model_list[name]['n_layers'], 
        model_list[name]['n_heads'], model_list[name]['d_ff'], 10000
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch = torch.randint(low=0, high=10000, size=(4, context_len), device=device)
    torch.cuda.synchronize()

    with nvtx.range("warmup"):
        for i in range(warmup):
            preds = model(batch)
            preds.mean().backward()
            torch.cuda.synchronize()

    with nvtx.range("full"):
        for i in range(n):
            preds = model(batch)
            preds.mean().backward()
            torch.cuda.synchronize()

def profile_full(context_len, name, warmup, n):
    model = BasicsTransformerLM(
        10000, context_len, model_list[name]['d_model'], model_list[name]['n_layers'], 
        model_list[name]['n_heads'], model_list[name]['d_ff'], 10000
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch = torch.randint(low=0, high=10000, size=(4, context_len), device=device)
    torch.cuda.synchronize()

    with nvtx.range("warmup"):
        for i in range(warmup):
            preds = model(batch)
            preds.mean().backward()
            optimizer.step()
            torch.cuda.synchronize()

    with nvtx.range("full"):
        for i in range(n):
            preds = model(batch)
            preds.mean().backward()
            optimizer.step()
            torch.cuda.synchronize()

def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

def annotated_scaled_dot_product_attention(Q, K, V):

    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

def profile_attn(context_len, name, warmup, n):

    d = model_list[name]['d_model']

    Q = torch.randn(4, context_len, d)
    K = torch.randn(4, context_len, d)
    V = torch.randn(4, context_len, d)

    for i in range(warmup + n):
        out = annotated_scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
    return 

if __name__ == '__main__':
    # sweep across the models and context lengths 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--len", type=int)
    args = parser.parse_args()

    profile_attn(args.len, args.model, 15, 30)