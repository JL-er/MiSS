import torch
from einops import rearrange
import triton
import triton.language as tl


@triton.autotune(
    configs=[
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
        triton.Config({'BM': 256}, num_stages=2, num_warps=2),
        triton.Config({'BM': 256}, num_stages=2, num_warps=4),
        triton.Config({'BM': 256}, num_stages=2, num_warps=8),
        triton.Config({'BM': 256}, num_stages=4, num_warps=2),
        triton.Config({'BM': 256}, num_stages=4, num_warps=4),
        triton.Config({'BM': 256}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bone_gradx(
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

    o_am = (i_m * BM + tl.arange(0, BM)) % M
    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)

    p_a = a + (o_am[:, None] * s_am + o_k[None, :] * s_ak)
    p_b = b + (o_k[ :, None] * s_bk + o_bn[None, :] * s_bn)

    p_bone = bone + o_k[ :, None] * s_bonem +  o_k[None, :] * s_bonen


    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        b_bone = tl.load(p_bone)
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)
        b_bone = tl.dot(b_bone,b_b, allow_tf32=False).to(b_b.dtype)+b_bone


        b_b = b_b+b_bone
        #b_acc = b_b
        # We accumulate along the K dimension.
        b_acc += tl.dot(b_a, b_b, allow_tf32=False)
        # Advance the ptrs to the next K block.
        p_a += BK * s_ak
        p_b += BK * s_bk
        p_bone += s_bonep

    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    #mask =  ( o_cm[:, None] < M) & (o_cn[None, :] < N )

    #b_c = b_acc

    p_c = c + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]

    tl.store(p_c, b_acc.to(c.dtype.element_ty) )

def bone_bwd(
    a: torch.Tensor,
    b: torch.Tensor,
    bone: torch.Tensor,

) -> torch.Tensor:
    #assert a.shape[2] == b.shape[1], 'Incompatible dimensions (A: {}x{}x{}, B: {}x{}x{})'.format(*a.shape, *b.shape)

    M, K = a.shape
    K, N = b.shape
    _,block,_ =bone.shape
    # Allocates output.
    c = a.new_empty(M, N)
    BK=BN = block


    def grid(meta): return (triton.cdiv(M, meta['BM']), triton.cdiv(N, BN))
    bone_gradx[grid](
        a, b, c, bone, 
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        bone.stride(0), bone.stride(1), bone.stride(2),
        BK=BK,BN=BN,G=4,
        ACTIVATION=None,
    )
    return c



# 使用方法
def bone(a,b, c):
    o = CustomEinsum.apply(a, b,c)
    return o

def gradx(do, ww):
    dx = torch.einsum('lo,do->ld', do, ww)
    return dx

import torch
from torch.autograd import gradcheck
import time
# 创建输入张量
dtype=torch.bfloat16
a = torch.randn((2048,64*40), dtype=dtype, requires_grad=True, device='cuda')

b = torch.randn((40,8,64,64), dtype=dtype, requires_grad=True, device='cuda')
c = torch.randn((8,64,64), dtype=dtype, requires_grad=True, device='cuda')
e = torch.randn((128,128), dtype=dtype, requires_grad=True, device='cuda')
do = torch.randn((2048,512), dtype=dtype, requires_grad=True, device='cuda')
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

w = b@c+c
ww= rearrange(w+b, 'a b r1 r2 ->(a r1) (b r2) ')

bb = rearrange(b, 'a b r1 r2 ->(a r1) (b r2) ')
dx_navie = gradx(do, ww)

dx = bone_bwd(do, bb.t(), c.transpose(1,2))

close = torch.allclose(dx_navie, dx, rtol=1e-05, atol=1e-08, equal_nan=False)
print('grad x',close)
print(dx_navie.reshape(-1))
print(dx.reshape(-1))

s = time.time()
for i in range(100):
    w = b@c+c
    ww= rearrange(w+b, 'a b r1 r2 ->(a r1) (b r2) ')
    dx_navie = gradx(do, ww)
torch.cuda.synchronize()
e = time.time()
print('bone navie ', (e-s)/100)

bbb = bb.t()
cc = c.transpose(1,2)
s = time.time()
for i in range(100):
    dx = bone_bwd(do, bbb, cc)
torch.cuda.synchronize()
e = time.time()
print('bone triton', (e-s)/100)
# dw = a.t()@do
# dw = rearrange(dw, '(a r1) (b r2) -> a b r1 r2 ', r1=64,r2=64)
# #dcc = torch.einsum('abjl,abjk->bkl', dw, b)
# # print('dc',dcc.reshape(-1))
# #tc  = torch.sum(b.transpose(2,3)@dw, dim=0)
# tc = (b.transpose(2,3)@dw)[-1]
# #tc = torch.sum(dw, dim=0)
# print(tc.reshape(-1))
# print(dw[0].reshape(-1))
# print(dw[1].reshape(-1))

# bb = rearrange(b, 'a b r1 r2 ->(a r1) (b r2) ')
# aa = a.t()
# dccc = bone_bwd_wb(aa, do, bb)
# print(dccc.reshape(-1))

