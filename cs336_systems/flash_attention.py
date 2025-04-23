import torch 
import numpy as np 
from einops import rearrange, einsum, reduce

import os
# os.environ["TRITON_INTERPRET"] = "1"

import triton
import triton.language as tl


def manual_backward(Q, K, V, O, dO, L):
    return 

class FlashAttentionTorch(torch.autograd.Function):

    compiled_backward = torch.compile(manual_backward)
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal):

        B_q = 16
        B_k = 8

        if len(Q.shape) == 2: Q = Q.unsqueeze(0)
        if len(K.shape) == 2: K = K.unsqueeze(0)
        if len(V.shape) == 2: V = V.unsqueeze(0)

        d = Q.shape[1]

        Q = rearrange(Q, "batch (T_q B_q) d -> T_q batch B_q d", B_q = B_q)
        K = rearrange(K, "batch (T_k B_k) d -> T_k batch B_k d", B_k = B_k)
        V = rearrange(V, "batch (T_k B_k) d -> T_k batch B_k d", B_k = B_k)

        d = Q.shape[-1]

        T_q = Q.shape[0]
        T_k = K.shape[0]

        assert K.shape == V.shape 
        assert Q.shape[1] == K.shape[1]

        batch = Q.shape[1]

        O = torch.empty((T_q, batch, B_q, d))
        L = torch.empty((T_q, batch, B_q))

        for i in range(T_q):

            # Load Q_i from global memory 
            Q_i = Q[i]

            O_i = torch.empty(batch, B_q, d)
            l_i = torch.empty(batch, B_q)
            m_i = torch.empty(batch, B_q)

            O_i[:, :, :] = 0 
            l_i[:, :] = 0
            m_i[:, :] = -1 * float('inf')

            for j in range(1, T_k + 1):
                # Load K_j, V_j from global memory 
                K_j = K[j-1] # shape batch x B_k x d
                V_j = V[j-1] 

                S_ij = einsum(Q_i, K_j, "batch B_q d, batch B_k d -> batch B_q B_k") / np.sqrt(d) 

                # if is_causal:
                #     # mask is B_q x B_k of 0 1 1 1... 0 0 1 1 ... 
                #     # query offset is query_tile_index * Q_TILE_SIZE 
                #     # key offset is (j - 1) * K_TILE_SIZE
                #     # mask[s][t] = 1 if s + query_tile_index * Q_TILE_SIZE < t + (j - 1) * K_TILE_SIZE
                #     mask = torch.zeros((B_q, B_k), dtype=torch.float32)
                #     for s in range(B_q):
                #         for t in range(B_k):
                #             if s + i * B_q < t + (j - 1) * B_k: 
                #                 mask[s,t] = 1
                #     S_ij = S_ij + torch.where(mask, 0, -1.0e6)

                rowmax = reduce(S_ij, "batch B_q B_k -> batch B_q", 'max') 
                m_ij = torch.maximum(m_i[:, :], rowmax) 
                assert S_ij.shape == (batch, B_q, B_k) 
                assert m_ij.shape == (batch, B_q) 

                P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1)) 
                assert P_ij.shape == (batch, B_q, B_k) 

                l_i = torch.exp(m_i[:, :] - m_ij) * l_i[:, :] + reduce(P_ij, "batch B_q B_k -> batch B_q", 'sum') 
                assert l_i.shape == (batch, B_q) 

                O_i = einsum(
                    torch.diag_embed(torch.exp(m_i[:, :] - m_ij)), 
                    O_i[:, :, :], 
                    "batch B_q B_q, batch B_q d -> batch B_q d"
                )
                O_i += einsum(
                    P_ij, V_j, 
                    "batch B_q B_k, batch B_k d -> batch B_q d"
                )

                # fill in 
                # O_i[:, :, :] = O_ij 
                # l_i[:, :] = l_ij 
                # m_i[:, :] = m_ij 
                m_i = m_ij

            j = T_k
            
            # Write O_i, L_i to global memory 
            O[i, :, :, :] = einsum(
                torch.linalg.inv(torch.diag_embed(l_i[:, :])), O_i[:, :, :], 
                'batch B_q B_q, batch B_q d -> batch B_q d'
            )
            L[i, :, :] = m_i[:, :] + torch.log(l_i[:, :])
                
        # reshape O, L
        O = rearrange(O, 'T_q batch B_q d -> batch (T_q B_q) d')
        L = rearrange(L, 'T_q batch B_q -> batch (T_q B_q)')
        
        ctx.save_for_backward(L)
        return O #, L
    
    @staticmethod
    def backward(ctx, Q, K, V, O, dO, L):
        
        return ctx.compiled_backward(Q, K, V, O, dO, L)


# @triton.jit
# def fill_diagonal(
#     diag,            # the diagonal matrix (Triton tensor)
#     values,          # values to fill the diagonal with (Triton tensor)
#     N,               # size of the square matrix
# ):
#     # Iterate over the diagonal
#     for i in range(N):
#         # Set the value on the diagonal
#         diag[i, i] = values[i]

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_tile_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_tile_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_tile_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # same shape as Q
    O_tile_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_tile_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q_i from global memory 
    Q = tl.load(Q_tile_ptr)

    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)
    # T_k = N_KEYS

    # Initialize a buffer to write to
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) - float('inf') 
    S_ij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
    rowmax = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    for j in range(1, T_k+1): 
        # Load K_j, V_j from global memory 
        K_j = tl.load(K_tile_ptr)
        V_j = tl.load(V_tile_ptr)

        S_ij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        # print()
        S_ij = tl.dot(Q, tl.trans(K_j), acc=S_ij)
        S_ij *= scale

        if is_causal:
            # mask is B_q x B_k of 0 1 1 1... 0 0 1 1 ... 
            # query offset is query_tile_index * Q_TILE_SIZE 
            # key offset is (j - 1) * K_TILE_SIZE
            # mask[s][t] = 1 if s + query_tile_index * Q_TILE_SIZE < t + (j - 1) * K_TILE_SIZE

            mask = (query_tile_index * Q_TILE_SIZE + tl.arange(0,Q_TILE_SIZE))[:, None] >= ((j-1) * K_TILE_SIZE + tl.arange(0,K_TILE_SIZE))[None, :]

            # mask = torch.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=torch.float32)
            # for s in range(Q_TILE_SIZE):
            #     for t in range(K_TILE_SIZE):
            #         if s + query_tile_index * Q_TILE_SIZE < t + (j - 1) * K_TILE_SIZE: 
            #             mask[s,t] = 1
            S_ij = S_ij + tl.where(mask, 0, -1.0e6)

            # S = torch.where(
            #     torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[None, None, :],
            #     S,
            #     -1e6
            # )

        rowmax = tl.max(S_ij, axis=-1) # 1?

        m_ij = tl.maximum(m, rowmax) 

        P_ij = tl.exp(S_ij - m_ij[:, None]) 

        l = l * tl.exp(m - m_ij) + tl.sum(P_ij, axis=-1) # 1
        
        diag = tl.exp(m - m_ij)
        O = O * diag[:, None]
        P_ij = P_ij.to(V_j.dtype)
        O = tl.dot(P_ij, V_j, acc=O)

        K_tile_ptr = K_tile_ptr.advance((K_TILE_SIZE,0))
        V_tile_ptr = V_tile_ptr.advance((K_TILE_SIZE,0))

        m = m_ij 

    # tl.device_print("m", m)
    # tl.device_print("rowmax", rowmax)
    # tl.device_print("m", m)

    O = O / l[:, None]
    
    tl.store(O_tile_ptr, 
             O)
    
    tl.store(L_tile_ptr,
             m + tl.log(l))    


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        Q.to(device)
        K.to(device)
        V.to(device)

        Q_TILE_SIZE = 32
        K_TILE_SIZE = 16

        if len(Q.shape) == 2: Q = Q.unsqueeze(0)
        if len(K.shape) == 2: K = K.unsqueeze(0)
        if len(V.shape) == 2: V = V.unsqueeze(0)

        batch_size = Q.shape[0]
        
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        D = Q.shape[-1]

        # D1 = Q.shape[1]
        # D2 = Q.shape[2]

        # T_q = tl.cdiv(D1, Q_TILE_SIZE)
        # T_k = tl.cdiv(D2, K_TILE_SIZE)

        T_q = N_QUERIES // Q_TILE_SIZE
        T_k = N_KEYS // K_TILE_SIZE

        # T_q = D2 // Q_TILE_SIZE
        # T_k = D2 // K_TILE_SIZE

        O = torch.empty((batch_size, N_QUERIES, D), dtype=torch.float32).to(device)
        L = torch.empty((batch_size, N_QUERIES), dtype=torch.float32).to(device)

        # launch grid: (T_q, batch_size)

        flash_fwd_kernel[(T_q, batch_size)](
            Q, K, V, 
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1), 
            N_QUERIES, N_KEYS, 
            scale=1/np.sqrt(D),
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )
                
        # reshape O, L
        # O = rearrange(O, 'T_q batch B_q d -> batch (T_q B_q) d')
        # L = rearrange(L, 'T_q batch B_q -> batch (T_q B_q)')

        # might need to reshape O, L

        # O = torch.zeros((batch_size, D1, D2)).to(device)
        # L = torch.zeros((batch_size, D1)).to(device)
        
        ctx.save_for_backward(L)
        ctx.is_causal = is_causal
        return O #, L
