import torch
from einops import rearrange
import triton
import triton.language as tl

@triton.jit
def bone_fwd_kernel(
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

    p_bone = bone + i_n * s_bonep + o_k[ :, None] * s_bonem +  o_k[None, :] * s_bonen
    b_bone = tl.load(p_bone)


    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)
        b_b = tl.dot(b_b, b_bone).to(b_b.dtype)+b_bone+b_b
        #b_acc = b_b
        # We accumulate along the K dimension.
        b_acc += tl.dot(b_a, b_b, allow_tf32=False)
        # Advance the ptrs to the next K block.
        p_a += BK * s_ak
        p_b += BK * s_bk

    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    #mask =  ( o_cm[:, None] < M) & (o_cn[None, :] < N )

    b_c = b_acc

    p_c = c + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]

    tl.store(p_c, b_c.to(c.dtype.element_ty) )


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
        b_bone = tl.dot(b_bone,b_b).to(b_b.dtype)+b_bone


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

    b_c = b_acc

    p_c = c + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]

    tl.store(p_c, b_c.to(c.dtype.element_ty) )


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
        dc += tl.dot(b_w.T, b_dw.to(b_w.dtype), allow_tf32=False)+b_dw
        #dc += tl.dot(b_w.T, b_dw, allow_tf32=False)+b_dw.to(tl.float32)


    p_c = c + o_k[ :, None] * s_cm +  o_k[None, :] * s_cn + i_n*s_cp

    tl.store(p_c, dc.to(c.dtype.element_ty) )

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
    BM=64
    BK= 64
    BN = block


    grid= (triton.cdiv(M, BM), triton.cdiv(N, BN))
    bone_gradx[grid](
        a, b, c, bone, 
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        bone.stride(0), bone.stride(1), bone.stride(2),
        BM=BM,BK=BK,BN=BN,G=4,
        num_stages=1,
        ACTIVATION=None,
    )
    return c

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
    c = torch.empty((bone_g, bone_b, bone_b), dtype=a.dtype, device=a.device)
    BM=64
    BK=BN = 64
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
        BM=BM,BK=BK,BN=BN,G=4,
        num_stages=1,
        ACTIVATION=None,
    )
    return c



class BoneTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):

        M, K = x.shape
        K, N = w.shape
        # Allocates output.
        o = x.new_empty(M, N)
        BM=32
        BK=BN = 64

        def grid(meta): return (triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))
        bone_fwd_kernel[grid](
            x, w, o, b, 
            M, N, K,
            x.stride(0), w.stride(1),
            w.stride(0), w.stride(1),
            o.stride(0), o.stride(1),
            b.stride(0), b.stride(1), b.stride(2),
            BM=BM,BK=BK,BN=BN,G=4,
            num_stages=1,
            ACTIVATION=None,
        )
        # ww = rearrange(w, '(a r1) (b r2) -> a b r1 r2', r1 = 64, r2 = 64)@b+b
        # ww = rearrange(ww, 'a b r1 r2 ->(a r1) (b r2) ')
        # o = x@(ww+w)
        ctx.save_for_backward(x, w, b)
        return o

    @staticmethod
    def backward(ctx, do):
        x, w, b= ctx.saved_tensors
        bone_g,bone_b,_=b.shape
        dx = bone_bwd(do, w.t(), b.transpose(1,2))
        dc = bone_bwd_wb(x.t(), do, w,bone_g,bone_b)

        return dx, None, dc

# 使用方法
# def bone(a,b, c):
#     o = CustomEinsum.apply(a, b,c)
#     return o

def flash_bone(x,w,b):
    o = BoneTriton.apply(x,w,b)
    return o

def bone(x,w,b):
    ww = rearrange(w, '(a r1) (b r2) -> a b r1 r2', r1 = 64 , r2 = 64 )@b+b
    ww= rearrange(ww, 'a b r1 r2 ->(a r1) (b r2) ')+w
    return x@ww


import torch
from torch.autograd import gradcheck
#torch.manual_seed(49)

# 创建输入张量
dtype=torch.bfloat16
block_size = 64
L, input, ouput, bone_b = 512, 2560,2560 , 64
a = torch.randn((L,input), dtype=dtype, requires_grad=True, device='cuda')

b = torch.randn((input,ouput), dtype=dtype, requires_grad=True, device='cuda')
c = torch.randn((ouput//bone_b, bone_b,bone_b), dtype=dtype, requires_grad=True, device='cuda')
do = torch.randn((L,ouput), dtype=dtype, requires_grad=True, device='cuda')
# a1=a.clone()
# b1=b.clone()
# c1=c.clone()
# do1 = do.clone()
o = flash_bone(a,b,c)

o.backward(do)
da, a.grad = a.grad.clone(), None
dc, c.grad = c.grad.clone(), None


# w = rearrange(b, '(a r1) (b r2) -> a b r1 r2', r1 = 64 , r2 = 64 )@c+c
# ww= rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')+b
o1 = bone(a,b,c)
o1.backward(do)


bb = rearrange(b, '(a r1) (b r2) -> a b r1 r2 ', r1=64,r2=64)
dw = a.t()@do
dw = rearrange(dw, '(a r1) (b r2) -> a b r1 r2 ', r1=64,r2=64)

tc  = torch.sum(bb.transpose(2,3)@dw, dim=0)
tc += torch.sum(dw, dim=0)
print(tc.reshape(-1))

dccc = bone_bwd_wb(a.t(), do, b,ouput//bone_b,bone_b)
print(dccc.reshape(-1))

torch.testing.assert_close(tc, dccc, rtol=0, atol=1e-4)



print('forward', torch.allclose(o, o1, rtol=1e-05, atol=1e-08, equal_nan=False))
close = torch.allclose(da, a.grad, rtol=1e-05, atol=1e-08, equal_nan=False)
print('grad a',close)

# close = torch.allclose(db, b.grad, rtol=1e-05, atol=1e-8, equal_nan=False)
# print('grad b',close)

close = torch.allclose(dc, c.grad, rtol=1e-05, atol=1e-4, equal_nan=False)
print('grad c',close)
print(c.grad.reshape(-1))
print(dc.reshape(-1))

#print(torch.allclose(da, a.grad, rtol=1e-05, atol=1e-4, equal_nan=False))
#torch.testing.assert_close(da, a.grad, rtol=0, atol=1e-8)
torch.testing.assert_close(dc, c.grad, rtol=0, atol=1e-4)
