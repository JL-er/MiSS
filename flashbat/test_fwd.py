# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang

# code adapted from
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.utils import contiguous



# @triton.autotune(
#     configs=[
#         triton.Config({'BM': 64, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=2, num_warps=2),
#         triton.Config({'BM': 64, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=2, num_warps=4),
#         triton.Config({'BM': 64, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=2, num_warps=8),
#         triton.Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=2, num_warps=2),
#         triton.Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=2, num_warps=4),
#         triton.Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=2, num_warps=8),
#         triton.Config({'BM': 64, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=2),
#         triton.Config({'BM': 64, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
#         triton.Config({'BM': 64, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=8),
#         triton.Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=2),
#         triton.Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
#         # triton.Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=8),
#         # triton.Config({'BM': 256, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=2, num_warps=2),
#         # triton.Config({'BM': 256, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=2, num_warps=4),
#         # triton.Config({'BM': 256, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=2, num_warps=8),
#         # triton.Config({'BM': 256, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=2),
#         # triton.Config({'BM': 256, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
#         # triton.Config({'BM': 256, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=8),
#     ],
#     key=['M', 'N', 'K'],
# )

@triton.autotune(
    configs=[
        triton.Config({'BM': 64}, num_stages=3, num_warps=2),
        triton.Config({'BM': 64}, num_stages=3, num_warps=4),
        triton.Config({'BM': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128}, num_stages=3, num_warps=2),
        triton.Config({'BM': 128}, num_stages=3, num_warps=4),
        triton.Config({'BM': 128}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64}, num_stages=2, num_warps=2),
        triton.Config({'BM': 64}, num_stages=2, num_warps=4),
        triton.Config({'BM': 64}, num_stages=2, num_warps=8),
        triton.Config({'BM': 128}, num_stages=2, num_warps=2),
        triton.Config({'BM': 128}, num_stages=2, num_warps=4),
        triton.Config({'BM': 128}, num_stages=2, num_warps=8),
        triton.Config({'BM': 64}, num_stages=4, num_warps=2),
        triton.Config({'BM': 64}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128}, num_stages=4, num_warps=2),
        triton.Config({'BM': 128}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128}, num_stages=4, num_warps=8),
        # triton.Config({'BM': 256}, num_stages=2, num_warps=2),
        # triton.Config({'BM': 256}, num_stages=2, num_warps=4),
        # triton.Config({'BM': 256}, num_stages=2, num_warps=8),
        # triton.Config({'BM': 256}, num_stages=4, num_warps=2),
        # triton.Config({'BM': 256}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 256}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a,
    b,
    c,
    bone,
    # Matrix dimensions
    M,
    N,
    K,
    s_am,
    s_ak,
    s_bk,
    s_bn,
    s_cm,
    s_cn,
    s_bonep,
    s_bonem,
    s_bonen,
    # Meta-parameters
    #BP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    G: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    NM, NN = tl.num_programs(0), tl.num_programs(1)
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    i_m, i_n = tl.swizzle2d(i_m, i_n,  NM, NN, G)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `p_a` is a block of [BM, BK] pointers
    # `p_b` is a block of [BK, BN] pointers
    # See above `Pointer Arithmetic` section for details

    o_am = (i_m * BM + tl.arange(0, BM))
    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)

    p_a = a + (o_am[:, None] * s_am + o_k[None, :] * s_ak)
    p_b = b + (o_k[ :, None] * s_bk + o_bn[None, :] * s_bn)

    p_bone = bone + i_n * s_bonep + o_k[ :, None] * s_bonem +  o_k[None, :] * s_bonen
    b_bone = tl.load(p_bone)


    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)
        b_b += tl.dot(b_b, b_bone, allow_tf32=False).to(b_b.dtype)+b_bone
        #b_acc = b_b
        # We accumulate along the K dimension.
        b_acc += tl.dot(b_a, b_b, allow_tf32=False)
        # Advance the ptrs to the next K block.
        p_a += BK * s_ak
        p_b += BK * s_bk

    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    mask = (o_cn[None, :] < N )

    #b_c = b_acc

    p_c = c + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]

    tl.store(p_c, b_acc.to(c.dtype.element_ty), mask=mask)





#@contiguous
def bone_fwd(
    bone: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    #assert a.shape[2] == b.shape[1], 'Incompatible dimensions (A: {}x{}x{}, B: {}x{}x{})'.format(*a.shape, *b.shape)

    B, L, K = a.shape
    M = B*L
    K, N = b.shape
    # Allocates output.
    c = a.new_empty(B, L, N)
    # print(c.shape,c.dtype)
    # print(N//64)
    # BM=64
    BK=BN = 64

    #grid=(triton.cdiv(M, BM), triton.cdiv(N, BN))
    def grid(meta): return (triton.cdiv(M, meta['BM']), triton.cdiv(N, BN))
    matmul_kernel[grid](
        a, b, c, bone, 
        M, N, K,
        a.stride(1), a.stride(2),
        b.stride(0), b.stride(1),
        c.stride(1), c.stride(2),
        bone.stride(0), bone.stride(1), bone.stride(2),
        BK=BK,BN=BN,G=4,
        ACTIVATION=None,
    )
    return c

# class BoneFn(torch.autograd.Function):

#     @staticmethod
#     #@contiguous
#     def forward(ctx, x, w, bone):
#         M, K = x.shape
#         K, N = w.shape
#         # Allocates output.
#         o = x.new_empty(M, N)
#         # print(c.shape,c.dtype)
#         # print(N//64)
#         def grid(meta): return (triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))
#         matmul_kernel[grid](
#             x, w, o, bone, 
#             M, N, K,
#             x.stride(0), x.stride(1),
#             w.stride(0), w.stride(1),
#             o.stride(0), o.stride(1),
#             bone.stride(0), bone.stride(1), bone.stride(2),
#             ACTIVATION=None,
#         )
#         ctx.save_for_backward(o)
#         return o
    
#     @staticmethod
#     def backward(ctx, do):


        

import time
#torch.manual_seed(0)
dtype = torch.bfloat16
B = 10
L = 1024
a = torch.randn((B, L,2048),device='cuda', dtype=dtype)
b = torch.randn((2048,4096),device='cuda', dtype=dtype)
c = torch.randn((64,64,64),device='cuda', dtype=dtype)

lora_a = torch.randn((32,4096),device='cuda', dtype=dtype)
lora_b = torch.randn((2048,32),device='cuda', dtype=dtype)

# c = debug(a, input, a)
# print("Debug output:", debug_output.cpu().numpy())
from einops import rearrange

xx = bone_fwd(c, a, b)
print(xx.reshape(-1))
# d = a@b+c
# print(d.shape, d)
w = rearrange(b, '(a r1) (b r2) -> a b r1 r2', r1 = 64, r2 = 64)@c+c
w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
e = a@(w+b)
print(e.reshape(-1))
close = torch.allclose(xx, e, rtol=1e-03, atol=1e-08, equal_nan=False)
print(close)
#torch.testing.assert_close(xx, e, rtol=1e-03, atol=1e-8)

# e = torch.addbmm(c,b,a)
# print(e)

def bone_flops(L,D,O,a,b,block):
    f = a*b*block*block*(2*block-1)
    x = L*O*(2*D-1)
    return f+x

ff = bone_flops(B*L, 2048, 4096, 32, 64, 64)

s = time.time()
for i in range(100):
    w = rearrange(b, '(a r1) (b r2) -> a b r1 r2', r1 = 64, r2 = 64)@c+c
    w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
    xxx = a@w
torch.cuda.synchronize()
e = time.time()
print('bone navie ', ff/(e-s)/100/1000)

s = time.time()
for i in range(100):
    xxx = a@(b+lora_b@lora_a)
torch.cuda.synchronize()
e = time.time()
#print('lora       ', ff/(e-s)/100)

s = time.time()
for i in range(100):
    xxx = bone_fwd(c, a, b)
torch.cuda.synchronize()
e = time.time()
print('bone triton', ff/(e-s)/100/1000)

# a = torch.randn(3,4,4)
# b = torch.randn(4,4)
# w = rearrange(b, '(a r1) (b r2) -> a b r1 r2', r1 = 64, r2 = 64)
# w = torch.einsum('abjk,bkl->abjl', [w, c])
# print(w)