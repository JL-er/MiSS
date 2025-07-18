import torch
from einops import rearrange
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BK': 32}, num_stages=2, num_warps=2),
        triton.Config({'BK': 32}, num_stages=2, num_warps=4),
        triton.Config({'BK': 32}, num_stages=2, num_warps=8),
        triton.Config({'BK': 32}, num_stages=4, num_warps=2),
        triton.Config({'BK': 32}, num_stages=4, num_warps=4),
        triton.Config({'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BK': 64}, num_stages=1, num_warps=2),
        triton.Config({'BK': 64}, num_stages=1, num_warps=4),
        triton.Config({'BK': 64}, num_stages=1, num_warps=8),
        triton.Config({'BK': 64}, num_stages=2, num_warps=2),
        triton.Config({'BK': 64}, num_stages=2, num_warps=4),
        triton.Config({'BK': 64}, num_stages=2, num_warps=8),
        triton.Config({'BK': 128}, num_stages=2, num_warps=2),
        triton.Config({'BK': 128}, num_stages=2, num_warps=4),
        triton.Config({'BK': 128}, num_stages=2, num_warps=8),
        triton.Config({'BK': 64}, num_stages=4, num_warps=2),
        triton.Config({'BK': 64}, num_stages=4, num_warps=4),
        triton.Config({'BK': 64}, num_stages=4, num_warps=8),
        triton.Config({'BK': 128}, num_stages=4, num_warps=2),
        triton.Config({'BK': 128}, num_stages=4, num_warps=4),
        triton.Config({'BK': 128}, num_stages=4, num_warps=8),
        triton.Config({'BK': 256}, num_stages=2, num_warps=2),
        triton.Config({'BK': 256}, num_stages=2, num_warps=4),
        triton.Config({'BK': 256}, num_stages=2, num_warps=8),
        triton.Config({'BK': 256}, num_stages=4, num_warps=2),
        triton.Config({'BK': 256}, num_stages=4, num_warps=4),
        triton.Config({'BK': 256}, num_stages=4, num_warps=8),
    ],
    key=['M','K','N'],
)

@triton.jit
def bone_gradwb(
    # Pointers to matrices
    a,
    b,
    c,
    w,
    # Matrix dimensions
    M,
    N,
    K,
    s_am,
    s_ak,
    s_bk,
    s_bn,
    s_cp,
    s_cm,
    s_cn,
    s_wm,
    s_wn,

    # Meta-parameters
    #BP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    G: tl.constexpr,
    ACTIVATION: tl.constexpr,
):

    i_n = tl.program_id(0)


    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)
    o_m = tl.arange(0, BM)
    o_block=tl.arange(0, 64)

    p_a = a + (o_m[:, None] * s_am + o_k[None, :] * s_ak)
    p_b = b + (o_k[ :, None] * s_bk + o_bn[None, :] * s_bn)

    p_w = w + s_wm * o_m[:, None] + s_wn * o_bn[None, :]

    #p_bone = bone + o_k[ :, None] * s_bonem +  o_k[None, :] * s_bonen + i_n * s_bonep
    dc = tl.zeros((64, 64), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BM)):
        b_dw = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):

            b_a = tl.load(p_a, mask=(o_k[None, :] < K - k * BK)&(o_m[:, None] < M - m * BM), other=0.0)
            b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)

            b_dw += tl.dot(b_a, b_b, allow_tf32=False)
            # Advance the ptrs to the next K block.
            p_a += BK * s_ak
            p_b += BK * s_bk

        b_w = tl.load(p_w, mask=(o_m[:, None] < M - m * BM), other=0.0)
        p_a += BM * s_am
        p_w += BM * s_wm
        p_a -= K * s_ak
        p_b -= K * s_bk
        dc += tl.dot(b_w.T, b_dw.to(b_w.dtype), allow_tf32=False)
        dc += b_dw


    p_c = c + o_block[ :, None] * s_cm +  o_block[None, :] * s_cn + i_n*s_cp

    tl.store(p_c, dc.to(c.dtype.element_ty) )




def bone_bwd_wb(
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    bone_g: int,
    bone_b: int,

) -> torch.Tensor:
    #assert a.shape[2] == b.shape[1], 'Incompatible dimensions (A: {}x{}x{}, B: {}x{}x{})'.format(*a.shape, *b.shape)

    M, K = a.shape
    K, N = b.shape
    #GN,block,_ =bone.shape
    # Allocates output.
    c = torch.zeros((bone_g, bone_b, bone_b), dtype=a.dtype, device=a.device)
    BM = BN = bone_b
    # print(c.shape,c.dtype)
    # print(N//64)
    #GB = triton.cdiv(N, BN)
    grid= (triton.cdiv(N, BN),)
    bone_gradwb[grid](
        a, b, c, w, 
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        w.stride(0), w.stride(1),
        BM=BM,BN=BN,G=4,
        ACTIVATION=None,
    )
    return c

# 使用方法
def bone(a,b, c):
    o = CustomEinsum.apply(a, b,c)
    return o


import torch
from torch.autograd import gradcheck
#torch.manual_seed(49)

# 创建输入张量
dtype=torch.float32
a = torch.randn((512,64*40), dtype=dtype, requires_grad=True, device='cuda')

b = torch.randn((40,40,64,64), dtype=dtype, requires_grad=True, device='cuda')
c = torch.randn((40,64,64), dtype=dtype, requires_grad=True, device='cuda')
e = torch.randn((128,128), dtype=dtype, requires_grad=True, device='cuda')
do = torch.randn((512,2560), dtype=dtype, requires_grad=True, device='cuda')
do1 = do.clone()
# o = bone(a,b,c)
# #o = torch.einsum('abjk,bkl->abjl', b, c)
# o.backward(do)
# print(o.reshape(-1))
# da, a.grad = a.grad.clone(), None
# db, b.grad = b.grad.clone(), None
# dc, c.grad = c.grad.clone(), None

# w = b@c+c
# ww= rearrange(w+b, 'a b r1 r2 ->(a r1) (b r2) ')
# o1 = a@ww
# o1.backward(do1)

# dw = a.t()@do
# dw = rearrange(dw, '(a r1) (b r2) -> a b r1 r2 ', r1=64,r2=64)
# #dcc = torch.einsum('abjl,abjk->bkl', dw, b)
# # print('dc',dcc.reshape(-1))
# #tc  = torch.sum(b.transpose(2,3)@dw, dim=0)
# #tc = (b.transpose(2,3)@dw)[-1]
# tc = torch.sum(dw, dim=0)
# print(tc.reshape(-1))
# # print(dw[0].reshape(-1))
# # print(dw[1].reshape(-1))

# bb = rearrange(b, 'a b r1 r2 ->(a r1) (b r2) ')
# aa = a.t()
# dccc = bone_bwd_wb(aa, do, bb)
# print(dccc.reshape(-1))

# close = torch.allclose(tc, dccc, rtol=1e-05, atol=1e-08, equal_nan=False)
# print('grad a',close)
# torch.testing.assert_close(tc, dccc, rtol=0, atol=1e-4)


dw = a.transpose(0,1)@do
dw = rearrange(dw, '(a r1) (b r2) -> a b r1 r2 ', r1=64,r2=64)
#dw = rearrange(dw, 'a r1 (b r2) -> a b r1 r2 ', r2=64)
tc  = torch.sum(b.transpose(2,3)@dw, dim=0)
tc += torch.sum(dw, dim=0)
#tc = dw
print(tc.reshape(-1))
bb = rearrange(b, 'a b r1 r2 ->(a r1) (b r2) ')
# dcc = torch.zeros((40,64,64), dtype=dtype, device='cuda')
# for i in range(a.size(0)):

#     aa = a[i].t()
#     dccc = bone_bwd_wb(aa, do, bb, 40 ,64)
#     dcc += dccc
#     close = torch.allclose(dw[i], dccc, rtol=1e-05, atol=1e-08, equal_nan=False)
#     print('grad a',close)
#     print(dw[i].reshape(-1))
#     print(dccc.reshape(-1))
#aa = rearrange(a, 'a b c ->b (a c) ')
dcc = bone_bwd_wb(a.t(), do, bb, 40 ,64)

close = torch.allclose(tc, dcc.to(tc.dtype), rtol=1e-05, atol=1e-08, equal_nan=False)
print('grad sum',close)
print(dcc.to(tc.dtype).reshape(-1))

torch.testing.assert_close(tc, dcc.to(tc.dtype), rtol=1e-3, atol=1e-8)
