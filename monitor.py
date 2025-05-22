import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import math
import matplotlib.pyplot as plt
import random
import json
import os
from datetime import datetime
import time

# --- Set Random Seeds for Reproducibility ---
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Configuration Parameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 64
OUTPUT_DIM = 64
HIDDEN_DIM = 64  # Hidden dimension for base model

# Method specific parameters (manually set for each method)
LORA_RANK = 2
VERA_RANK = 64
BLOCK_DIAG_RANK = 4
PISSA_RANK = 2
DORA_RANK = 1
PROLORA_RANK = 2
ADALORA_RANK = 2
MOS_RANK = 2

# Training parameters
BASE_LR = 0.001  # Learning rate for base model
ADAPT_LR = 0.001  # Learning rate for adaptation methods
BASE_EPOCHS = 250  # Epochs for base model training
ADAPT_EPOCHS = 100  # Epochs for adaptation methods
EVAL_INTERVAL = 10  # Evaluation interval
N_TRAIN = 50  # Number of training data points
N_VALID = 100  # Number of validation data points
NOISE_STD = 0.05  # Noise standard deviation for training data

# --- Data Generation Functions ---
def base_function(x):
    """Original function to fit with the base model"""
    return torch.sin(2 * torch.pi * x)

def modified_function(x):
    """Modified function for adaptation (slightly different)"""
    return torch.sin(2 * torch.pi * x) + 0.3 * torch.cos(3 * torch.pi * x)

def generate_data(n_samples, func, noise_std, device):
    """Generate synthetic data from the given function"""
    x = torch.rand(n_samples, INPUT_DIM) * 2 - 1  # X in range [-1, 1]
    y = func(x) + torch.randn(n_samples, OUTPUT_DIM) * noise_std
    return x.to(device), y.to(device)

# Generate datasets
x_train_base, y_train_base = generate_data(N_TRAIN, base_function, NOISE_STD, DEVICE)
x_valid_base, y_valid_base = generate_data(N_VALID, base_function, 0.0, DEVICE)  # No noise for validation

x_train_adapt, y_train_adapt = generate_data(N_TRAIN, modified_function, NOISE_STD, DEVICE)
x_valid_adapt, y_valid_adapt = generate_data(N_VALID, modified_function, 0.0, DEVICE)  # No noise for validation

# --- Base MLP Model ---
class BaseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.adapter_name = "BaseModel"

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# --- Common: Freeze Original Layer ---
def freeze_original_layer(original_layer):
    for param in original_layer.parameters():
        param.requires_grad = False

# --- LoRA Adapter ---
class LoRAAdapter(nn.Module):
    def __init__(self, original_layer, rank, scale=1.0):
        super().__init__()
        self.original_layer = original_layer
        # Freeze the original layer
        freeze_original_layer(original_layer)
            
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.scale = scale
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B initialized to zero

    def forward(self, x):
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA adaptation: W = W0 + BA
        delta = self.lora_B @ self.lora_A @ x.T
        return original_output + self.scale * delta.T

    def num_adapter_parameters(self):
        return self.lora_A.numel() + self.lora_B.numel()

# --- VeRA Adapter ---
class VeRAAdapter(nn.Module):
    def __init__(self, original_layer, rank, d_init_val=0.1):
        super().__init__()
        self.original_layer = original_layer
        # Freeze the original layer
        freeze_original_layer(original_layer)
            
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        
        # Frozen pseudo-shared matrices
        self.A_frozen = torch.empty(rank, self.in_features)
        self.B_frozen = torch.empty(self.out_features, rank)
        
        nn.init.kaiming_uniform_(self.A_frozen, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B_frozen, a=math.sqrt(5))
        
        self.A_frozen = self.A_frozen.to(DEVICE)
        self.B_frozen = self.B_frozen.to(DEVICE)
        self.A_frozen.requires_grad = False
        self.B_frozen.requires_grad = False
        
        # Trainable scaling vectors
        self.b_vec = nn.Parameter(torch.zeros(self.out_features))  # Output dimension
        self.d_vec = nn.Parameter(torch.full((rank,), d_init_val))  # Rank dimension

    def forward(self, x):
        # Original output
        original_output = self.original_layer(x)
        
        # VeRA adaptation: W = W0 + diag(b) * B * diag(d) * A
        Lambda_b = torch.diag(self.b_vec)
        Lambda_d = torch.diag(self.d_vec)
        
        delta = Lambda_b @ self.B_frozen @ Lambda_d @ self.A_frozen @ x.T
        return original_output + delta.T

    def num_adapter_parameters(self):
        return self.b_vec.numel() + self.d_vec.numel()

# --- Our Method: Block Diagonal Adapter ---
class BlockDiagonalAdapter(nn.Module):
    def __init__(self, original_layer, rank): # rank 参数现在代表 ShardLoRA 的秩 r
        super().__init__()
        self.original_layer = original_layer
        freeze_original_layer(original_layer) # 假设此函数存在或在外部调用

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.r = rank # 这是论文 [cite: 110, 111] 中的 'r' 或 Algorithm 1 中的 self.r

        if self.in_features % self.r != 0:
            raise ValueError(f"Input dimension ({self.in_features}) "
                             f"must be divisible by shard_lora_rank ({self.r}) "
                             f"for this ShardLoRA-style adapter.")

        # 可训练矩阵 D (论文中表示为 D in R^(r x d)[cite: 109, 111], 或 Algorithm 1 中的 self.disha)
        # 大小为 (shard_lora_rank, out_features)
        self.disha_D = nn.Parameter(torch.empty(self.r, self.out_features))
        # 论文 [cite: 42] 中提到 ShardLoRA (Ours) 的 D 初始化为0
        nn.init.zeros_(self.disha_D)

    def forward(self, x): # x: (batch_size, ..., in_features)
        original_output = self.original_layer(x)

        # 输入聚合 S，参考论文 Section 4.2 [cite: 112, 119] 和 Algorithm 1 [cite: 110]
        # x 原始 shape: (batch_size, seq_len, in_features) 或 (batch_size, in_features)
        # 需要重塑为 (batch_size, ..., in_features // self.r, self.r)
        # 然后在倒数第二个维度上求和 (即 in_features // self.r 这个维度)
        
        leading_dims = x.shape[:-1] # (batch_size, seq_len) 或 (batch_size,)
        in_features_dim = x.shape[-1]

        if in_features_dim != self.in_features:
            raise ValueError(f"Input feature dimension {in_features_dim} does not match layer's in_features {self.in_features}")

        # Reshape for summation
        # x_reshaped: (batch_size, ..., self.in_features // self.r, self.r)
        x_reshaped_for_sum = x.view(*leading_dims, self.in_features // self.r, self.r)
        
        # Sum along the (self.in_features // self.r) dimension
        # S: (batch_size, ..., self.r)
        S = torch.sum(x_reshaped_for_sum, dim=-2)

        # 计算 delta_y = S @ D
        # S shape: (batch_size, ..., self.r)
        # self.disha_D shape: (self.r, self.out_features)
        # delta_y shape: (batch_size, ..., self.out_features)
        delta_y = torch.matmul(S, self.disha_D)

        return original_output + delta_y

    def num_adapter_parameters(self):
        # 参数量为 r * out_features [cite: 117, 124]
        return self.disha_D.numel()


# --- 1. PiSSA Adapter ---
class PiSSAAdapter(nn.Module):
    def __init__(self, original_layer, rank):
        super().__init__()
        self.original_layer_weights_only = original_layer # Keep a reference if needed for W_res
        freeze_original_layer(original_layer) # The original layer object itself might not be used if W_res is handled separately

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.adapter_name = f"PiSSA (Rank {rank})"

        # Perform SVD on original_layer.weight (W)
        # W = U S V^T
        # This is conceptual. In practice, you'd load W and decompose.
        W = original_layer.weight.data.clone() # Shape: (out_features, in_features)
        try:
            U, S_diag, Vh = torch.linalg.svd(W, full_matrices=False)
        except torch.linalg.LinAlgError: # Handle cases where SVD might fail for zero matrices etc.
            # Fallback or error handling
            print(f"SVD failed for layer, initializing PiSSA with LoRA-like approach as fallback.")
            # Fallback to LoRA-like initialization for A and B if SVD fails
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.W_res = nn.Parameter(W, requires_grad=False) # Freeze original weights
            self.bias = nn.Parameter(original_layer.bias.data.clone() if original_layer.bias is not None else None, requires_grad=False)
            self.svd_initialized = False
            return


        # Principal components for A and B
        U_r = U[:, :rank]
        S_r_diag = S_diag[:rank]
        Vh_r = Vh[:rank, :] # Vh is V.T, so Vh_r is (V_r)^T

        S_r_sqrt = torch.sqrt(S_r_diag)

        self.lora_A = nn.Parameter(torch.diag(S_r_sqrt) @ Vh_r) # S_r^(1/2) @ V_r^T
        self.lora_B = nn.Parameter(U_r @ torch.diag(S_r_sqrt))   # U_r @ S_r^(1/2)

        # Residual weights (frozen)
        if rank < S_diag.size(0) and rank < U.size(1) and rank < Vh.size(0):
            U_res = U[:, rank:]
            S_res_diag = S_diag[rank:]
            Vh_res = Vh[rank:, :]
            self.W_res = nn.Parameter(U_res @ torch.diag(S_res_diag) @ Vh_res, requires_grad=False)
        else: # If rank is too large, residual might be zero or very small
            self.W_res = nn.Parameter(torch.zeros_like(W), requires_grad=False)

        # Freeze original bias if it exists
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None
        self.svd_initialized = True


    def forward(self, x):
        # For PiSSA, we need to be careful with dimensions
        # The error occurs because we're trying to add tensors of incompatible shapes
        
        if not self.svd_initialized: # Fallback case
             original_output = torch.functional.F.linear(x, self.W_res, self.bias)
             delta = self.lora_B @ self.lora_A
             return original_output + torch.functional.F.linear(x, delta, None)

        # Residual part
        output_res = torch.functional.F.linear(x, self.W_res, self.bias)
        
        # Adaptable part - simplified approach using F.linear
        # Instead of manually handling matrix multiplications, use F.linear
        delta_W = self.lora_B @ self.lora_A
        adapt_output = torch.functional.F.linear(x, delta_W, None)
        
        return output_res + adapt_output

    def num_adapter_parameters(self):
        return self.lora_A.numel() + self.lora_B.numel()

# --- 2. DoRA Adapter ---
class DoRAAdapter(nn.Module):
    def __init__(self, original_layer, rank, lora_alpha=1.0): # lora_alpha is often rank
        super().__init__()
        self.original_layer = original_layer # Keep for W0 and bias
        freeze_original_layer(original_layer)

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.lora_alpha = lora_alpha # LoRA scaling factor
        self.adapter_name = f"DoRA (Rank {rank})"

        # Pre-trained weight W0 (direction component initially)
        self.W0 = original_layer.weight.data.clone()
        self.W0.requires_grad = False # Direction component of W0 is frozen

        # Magnitude vector m, initialized from ||W0||_c
        # Shape: (out_features, 1) to allow broadcasting for column norms
        # Or (1, in_features) if norms are taken row-wise (paper says column-wise for W in R^dxk)
        # Assuming W0 is (out_features, in_features), ||W0||_c means norm of each column vector
        # So m should correspond to columns of W0.
        # If W' = m * (V / ||V||_c), and W' is (out_features, in_features)
        # V is (out_features, in_features), ||V||_c is (1, in_features)
        # m should be (1, in_features) to scale columns
        # The paper Figure 1 shows m as (1 x k) for W (d x k) -> k = in_features
        self.m = nn.Parameter(torch.linalg.norm(self.W0, dim=0, keepdim=True)) # Norm along columns (dim=0)

        # LoRA matrices for directional update (delta_V = BA)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features)) # r x k
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank)) # d x r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = self.lora_alpha / self.rank

        self.bias = None
        if original_layer.bias is not None:
            self.bias = original_layer.bias.data.clone()
            self.bias.requires_grad = False


    def forward(self, x):
        # W_adapted = m * normalize(W0 + BA)
        # y = x @ W_adapted.T + bias_orig
        # or y = m * ( (x @ (W0+BA).T) / ||W0+BA||_c_row_wise ) + bias if m scales output

        # From paper eq (5): W' = m * (W0 + BA) / ||W0 + BA||_c
        # So, y = x @ W'.T + bias = x @ (m * (W0 + BA) / ||W0 + BA||_c).T + bias
        # This means m needs to be applied after normalization, or W' precomputed.

        delta_W = self.lora_B @ self.lora_A # d x k
        adapted_V = self.W0 + self.scaling * delta_W # Directional part update

        # Column-wise norm of adapted_V
        norm_adapted_V = torch.linalg.norm(adapted_V, dim=0, keepdim=True) + 1e-5 # Avoid division by zero

        # Normalized direction
        V_normalized = adapted_V / norm_adapted_V

        # Final adapted weight
        W_prime = self.m * V_normalized # Element-wise multiplication due to m being (1, k) and V_norm (d, k)

        return torch.functional.F.linear(x, W_prime, self.bias)

    def num_adapter_parameters(self):
        return self.m.numel() + self.lora_A.numel() + self.lora_B.numel()


# --- ProLoRA Adapter ---
class ProLoRAAdapter(nn.Module):
    def __init__(self, original_layer, rank, unshared_rank_u=1, sharing_ratio_m_A=None, sharing_ratio_n_B=None, base_stride_sA=1, base_stride_sB=1):
        super().__init__()
        self.original_layer = original_layer
        freeze_original_layer(original_layer)

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.r = rank
        self.u = unshared_rank_u # unshared rank
        
        # Auto-adjust sharing ratios based on input dimensions if not provided
        # Choose sharing ratios that divide the dimensions evenly
        if sharing_ratio_m_A is None:
            # Default to 1 if in_features < 2, otherwise use 2 if divisible, else 1
            self.m_A = 1 if self.in_features < 2 else (2 if self.in_features % 2 == 0 else 1)
        else:
            self.m_A = sharing_ratio_m_A
            
        if sharing_ratio_n_B is None:
            # Default to 1 if out_features < 2, otherwise use 2 if divisible, else 1
            self.n_B = 1 if self.out_features < 2 else (2 if self.out_features % 2 == 0 else 1)
        else:
            self.n_B = sharing_ratio_n_B
            
        self.sA = base_stride_sA
        self.sB = base_stride_sB
        self.adapter_name = f"ProLoRA (Rank {rank})"

        # Verify dimensions are compatible with sharing ratios
        if self.in_features % self.m_A != 0:
            print(f"Warning: in_features {self.in_features} not divisible by m_A {self.m_A}. Adjusting m_A to 1.")
            self.m_A = 1
            
        if self.out_features % self.n_B != 0:
            print(f"Warning: out_features {self.out_features} not divisible by n_B {self.n_B}. Adjusting n_B to 1.")
            self.n_B = 1

        if (self.r - self.u) <= 0 :
            print("Warning: No shared ranks in ProLoRA, behaves like LoRA with rank u.")
            self.is_lora_equivalent = True
            self.lora_A_eq = nn.Parameter(torch.empty(self.u, self.in_features))
            self.lora_B_eq = nn.Parameter(torch.empty(self.out_features, self.u))
            nn.init.kaiming_uniform_(self.lora_A_eq, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_eq)
            self.shared_rank = 0
        else:
            self.is_lora_equivalent = False
            self.shared_rank = self.r - self.u
            # Unshared parts
            self.A_u = nn.Parameter(torch.empty(self.u, self.in_features))
            self.B_u = nn.Parameter(torch.empty(self.out_features, self.u))
            nn.init.kaiming_uniform_(self.A_u, a=math.sqrt(5))
            nn.init.zeros_(self.B_u)

            # Shared base chunks (A0, B0)
            # Dimensions of A0: (shared_rank, in_features / m_A)
            # Dimensions of B0: (out_features / n_B, shared_rank)
            self.A0_chunk_in_dim = self.in_features // self.m_A
            self.B0_chunk_out_dim = self.out_features // self.n_B

            self.A0_shared = nn.Parameter(torch.empty(self.shared_rank, self.A0_chunk_in_dim))
            self.B0_shared = nn.Parameter(torch.empty(self.B0_chunk_out_dim, self.shared_rank))

            # Rectified Kaiming for A0, Zeros for B0
            # For A0, fan_in is the full in_features for the equivalent non-sharded matrix part
            gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5)) # kaiming_uniform default uses leaky_relu
            std_A0 = gain / math.sqrt(self.in_features) # Use full in_features for bound calculation
            bound_A0 = math.sqrt(3.0) * std_A0
            nn.init.uniform_(self.A0_shared, -bound_A0, bound_A0)
            nn.init.zeros_(self.B0_shared)

    def _get_rotated_chunks(self):
        # Construct full A_shared and B_shared from A0, B0 with rotation
        A_s_chunks = []
        for i in range(self.m_A):
            stride = i * self.sA
            # Roll along rank dimension (dim=0 for A0_shared)
            Ai_chunk = torch.roll(self.A0_shared, shifts=stride, dims=0)
            A_s_chunks.append(Ai_chunk)
        A_s = torch.cat(A_s_chunks, dim=1) # Concatenate along hidden_dim for A

        B_s_chunks = []
        for i in range(self.n_B):
            stride = i * self.sB
            # Roll along rank dimension (dim=1 for B0_shared)
            Bi_chunk = torch.roll(self.B0_shared, shifts=stride, dims=1)
            B_s_chunks.append(Bi_chunk)
        B_s = torch.cat(B_s_chunks, dim=0) # Concatenate along out_dim for B
        return A_s, B_s


    def forward(self, x):
        original_output = self.original_layer(x)

        if self.is_lora_equivalent:
            delta_W = self.lora_B_eq @ self.lora_A_eq
        else:
            A_s, B_s = self._get_rotated_chunks() # A_s: (shared_r, in_feat), B_s: (out_feat, shared_r)

            # Combine unshared and shared parts for delta_W calculation
            delta_W_unshared = self.B_u @ self.A_u
            delta_W_shared = B_s @ A_s
            delta_W = delta_W_unshared + delta_W_shared # Assuming this additive composition

        # Need to handle x dimensions for matmul correctly
        if x.ndim == 2: # (batch, features)
            adapt_output = torch.functional.F.linear(x, delta_W, None)
        elif x.ndim == 3: # (batch, seq_len, features)
            original_shape = x.shape
            x_reshaped = x.reshape(-1, x.shape[-1])
            adapt_output_reshaped = torch.functional.F.linear(x_reshaped, delta_W, None)
            adapt_output = adapt_output_reshaped.reshape(original_shape[0], original_shape[1], -1)
        else:
            raise ValueError("Input x must be 2D or 3D")

        return original_output + adapt_output


    def num_adapter_parameters(self):
        if self.is_lora_equivalent:
            return self.lora_A_eq.numel() + self.lora_B_eq.numel()

        params = self.A_u.numel() + self.B_u.numel()
        if self.shared_rank > 0 :
             params += self.A0_shared.numel() + self.B0_shared.numel()
        return params

# --- AdaLoRA Adapter ---
class AdaLoRAAdapter(nn.Module):
    def __init__(self, original_layer, initial_rank):
        super().__init__()
        self.original_layer = original_layer
        freeze_original_layer(original_layer)

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.r = initial_rank # Rank can change during training
        self.adapter_name = f"AdaLoRA (Rank {initial_rank})"

        # P Lambda Q parameterization
        # P: (out_features, r), Q: (r, in_features), Lambda: (r, r) diagonal
        self.lora_P = nn.Parameter(torch.empty(self.out_features, self.r))
        self.lora_Q = nn.Parameter(torch.empty(self.r, self.in_features))
        self.lora_Lambda_diag = nn.Parameter(torch.empty(self.r)) # Store diagonal of Lambda

        nn.init.normal_(self.lora_P, 0, 0.02) # Example: Gaussian init
        nn.init.normal_(self.lora_Q, 0, 0.02) # Example: Gaussian init
        nn.init.zeros_(self.lora_Lambda_diag) # Lambda initialized to zero

        self.bias = None
        if original_layer.bias is not None:
            self.bias = original_layer.bias.data.clone()
            self.bias.requires_grad = False

    def forward(self, x):
        original_output = self.original_layer(x)

        # Delta W = P @ diag(Lambda_diag) @ Q
        Lambda_matrix = torch.diag(self.lora_Lambda_diag)
        delta_W = self.lora_P @ Lambda_matrix @ self.lora_Q

        if x.ndim == 2:
            adapt_output = torch.functional.F.linear(x, delta_W, None)
        elif x.ndim == 3:
            original_shape = x.shape
            x_reshaped = x.reshape(-1, x.shape[-1])
            adapt_output_reshaped = torch.functional.F.linear(x_reshaped, delta_W, None)
            adapt_output = adapt_output_reshaped.reshape(original_shape[0], original_shape[1], -1)
        else:
            raise ValueError("Input x must be 2D or 3D")

        return original_output + adapt_output

    def num_adapter_parameters(self):
        # This is complex as r changes. This is for current r.
        return self.lora_P.numel() + self.lora_Q.numel() + self.lora_Lambda_diag.numel()

# --- 5. MoS (Mixture of Shards) Adapter ---
class MoSAdapter(nn.Module):
    def __init__(self, original_layer, rank_per_layer, shard_dim_ratio=2):
        super().__init__()
        self.original_layer = original_layer
        freeze_original_layer(original_layer)
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank_per_layer = rank_per_layer # Effective rank for this layer
        
        # For simplicity, use a configuration that works with our 1-dim inputs
        # Assuming rank is 4, shard_dim_ratio is 4, we'd have 4 shards
        num_shards_A = shard_dim_ratio
        num_shards_B = shard_dim_ratio
        num_selected_shards_A = shard_dim_ratio
        num_selected_shards_B = shard_dim_ratio
        
        # Dimensions for our shards - simplified for this implementation
        shard_dim_A_rank = rank_per_layer // num_selected_shards_A
        shard_dim_A_in = self.in_features
        shard_dim_B_out = self.out_features  
        shard_dim_B_rank = rank_per_layer // num_selected_shards_B
        
        self.num_selected_shards_A = num_selected_shards_A
        self.num_selected_shards_B = num_selected_shards_B

        # Simulate global shard pools (simplified for this implementation)
        self.A_shard_pool = nn.ParameterList(
            [nn.Parameter(torch.empty(shard_dim_A_rank, shard_dim_A_in)) for _ in range(num_shards_A)]
        )
        self.B_shard_pool = nn.ParameterList(
            [nn.Parameter(torch.empty(shard_dim_B_out, shard_dim_B_rank)) for _ in range(num_shards_B)]
        )

        # Initialization
        for shard_A in self.A_shard_pool:
            nn.init.kaiming_uniform_(shard_A, a=math.sqrt(5))
        for shard_B in self.B_shard_pool:
            nn.init.zeros_(shard_B)

        # Fixed selection for demonstration - use all shards
        self.selected_A_indices = list(range(num_selected_shards_A))
        self.selected_B_indices = list(range(num_selected_shards_B))

        self.bias = None
        if original_layer.bias is not None:
            self.bias = original_layer.bias.data.clone()
            self.bias.requires_grad = False

    def _construct_lora_matrices(self):
        # Get selected shards
        selected_A_shards = [self.A_shard_pool[i] for i in self.selected_A_indices]
        lora_A_eff = torch.cat(selected_A_shards, dim=0)  # Cat along rank dimension

        selected_B_shards = [self.B_shard_pool[i] for i in self.selected_B_indices]
        lora_B_eff = torch.cat(selected_B_shards, dim=1)  # Cat along rank dimension

        return lora_A_eff, lora_B_eff

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_A, lora_B = self._construct_lora_matrices()

        delta_W = lora_B @ lora_A

        if x.ndim == 2:
            adapt_output = torch.functional.F.linear(x, delta_W, None)
        elif x.ndim == 3:
            original_shape = x.shape
            x_reshaped = x.reshape(-1, x.shape[-1])
            adapt_output_reshaped = torch.functional.F.linear(x_reshaped, delta_W, None)
            adapt_output = adapt_output_reshaped.reshape(original_shape[0], original_shape[1], -1)
        else:
            raise ValueError("Input x must be 2D or 3D")

        return original_output + adapt_output

    def num_adapter_parameters(self):
        # Parameters are in the shard pools
        total_params = 0
        for p in self.A_shard_pool:
            total_params += p.numel()
        for p in self.B_shard_pool:
            total_params += p.numel()
        return total_params

# --- Training Functions ---
def train_base_model(model, x_train, y_train, x_valid, y_valid, epochs, lr, eval_interval):
    """Train the base model from scratch"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'valid_loss': [], 'epochs': []}
    
    print(f"\nTraining Base Model...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                valid_pred = model(x_valid)
                valid_loss = criterion(valid_pred, y_valid)
                history['train_loss'].append(loss.item())
                history['valid_loss'].append(valid_loss.item())
                history['epochs'].append(epoch + 1)
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Valid Loss: {valid_loss.item():.6f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    final_train_loss = loss.item()
    
    model.eval()
    with torch.no_grad():
        final_valid_loss = criterion(model(x_valid), y_valid).item()
    
    print(f"Base Model - Total Parameters: {total_params}, Final Train Loss: {final_train_loss:.6f}, Final Valid Loss: {final_valid_loss:.6f}")
    
    return model, history, total_params

def train_adapter(model, x_train, y_train, x_valid, y_valid, epochs, lr, eval_interval):
    """Train only the adapter parameters while keeping base model frozen"""
    # Only optimize trainable parameters (adapter parameters)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'valid_loss': [], 'epochs': []}
    
    # For detailed step-by-step logging
    detailed_history = {'step': [], 'epoch': [], 'train_loss': [], 'valid_loss': []}
    
    adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTraining {model.adapter_name} - Trainable Parameters: {adapter_params}")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        # Log every step
        model.eval()
        with torch.no_grad():
            valid_pred = model(x_valid)
            valid_loss = criterion(valid_pred, y_valid).item()
            train_loss = loss.item()
            detailed_history['step'].append(epoch)
            detailed_history['epoch'].append(epoch + 1)
            detailed_history['train_loss'].append(train_loss)
            detailed_history['valid_loss'].append(valid_loss)
        
        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                valid_pred = model(x_valid)
                valid_loss = criterion(valid_pred, y_valid)
                history['train_loss'].append(loss.item())
                history['valid_loss'].append(valid_loss.item())
                history['epochs'].append(epoch + 1)
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Valid Loss: {valid_loss.item():.6f}")
    
    final_train_loss = loss.item()
    
    model.eval()
    with torch.no_grad():
        final_valid_loss = criterion(model(x_valid), y_valid).item()
    
    print(f"{model.adapter_name} - Trainable Parameters: {adapter_params}, Final Train Loss: {final_train_loss:.6f}, Final Valid Loss: {final_valid_loss:.6f}")
    
    return history, adapter_params, final_train_loss, final_valid_loss, detailed_history

def calculate_parameter_counts(input_dim, hidden_dim, output_dim, lora_rank, vera_rank, block_diag_rank, pissa_rank, dora_rank, prolora_rank, adalora_rank, mos_rank):
    """Calculate parameter counts for different adaptation methods"""
    param_counts = {}
    
    # Base model total parameters
    base_model = BaseModel(input_dim, hidden_dim, output_dim)
    param_counts['base_model'] = sum(p.numel() for p in base_model.parameters())
    
    # Linear layer parameters (focus on first layer for adaptation)
    linear_layer = nn.Linear(input_dim, hidden_dim)
    param_counts['layer_weights'] = linear_layer.weight.numel()  # weights only
    param_counts['layer_total'] = sum(p.numel() for p in linear_layer.parameters())  # weights + bias
    
    # LoRA parameters
    param_counts['lora'] = lora_rank * input_dim + hidden_dim * lora_rank
    
    # VeRA parameters
    param_counts['vera'] = hidden_dim + vera_rank  # b_vec and d_vec
    
    # Block Diagonal parameters
    if block_diag_rank == 0: # Avoid division by zero if rank is 0 (though unlikely for this adapter)
        param_counts['block_diagonal'] = 0
    else:
        param_counts['block_diagonal'] = (hidden_dim * input_dim) // block_diag_rank
    
    # PiSSA parameters
    param_counts['pissa'] = pissa_rank * input_dim + hidden_dim * pissa_rank
    
    # DoRA parameters
    param_counts['dora'] = input_dim + dora_rank * input_dim + hidden_dim * dora_rank
    
    # ProLoRA parameters (simplified, assumes 1 shared dimension)
    unshared_rank = 1  # 1/4 of rank is unshared for this example
    shared_rank = prolora_rank - unshared_rank
    sharing_ratio = 2  # m_A and n_B are both 2
    param_counts['prolora'] = (unshared_rank * input_dim + hidden_dim * unshared_rank) + \
                             (shared_rank * (input_dim // sharing_ratio) + (hidden_dim // sharing_ratio) * shared_rank)
    
    # AdaLoRA parameters
    param_counts['adalora'] = hidden_dim * adalora_rank + adalora_rank * input_dim + adalora_rank  # P, Q, Lambda_diag
    
    # MoS parameters
    shard_ratio = 2  # Number of shards
    param_counts['mos'] = (mos_rank // shard_ratio) * input_dim * shard_ratio + hidden_dim * (mos_rank // shard_ratio) * shard_ratio
    
    return param_counts

def run_experiment():
    """Run the complete experiment with base model training and adaptation methods"""
    print(f"Running experiments on device: {DEVICE}")
    
    # Create directory for results if it doesn't exist
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_json_path = os.path.join(results_dir, f"adaptation_results_{timestamp}.json")
    
    # Dictionary to store all experiment data
    experiment_data = {
        "config": {
            "device": str(DEVICE),
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": OUTPUT_DIM,
            "seed": SEED,
            "base_epochs": BASE_EPOCHS,
            "adapt_epochs": ADAPT_EPOCHS,
            "base_lr": BASE_LR,
            "adapt_lr": ADAPT_LR,
            "noise_std": NOISE_STD,
            "n_train": N_TRAIN,
            "n_valid": N_VALID
        },
        "method_configs": {
            "lora_rank": LORA_RANK,
            "vera_rank": VERA_RANK,
            "block_diag_rank": BLOCK_DIAG_RANK,
            "pissa_rank": PISSA_RANK,
            "dora_rank": DORA_RANK,
            "prolora_rank": PROLORA_RANK,
            "adalora_rank": ADALORA_RANK,
            "mos_rank": MOS_RANK
        },
        "results": {}
    }
    
    # Calculate and display parameter counts
    param_counts = calculate_parameter_counts(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
        LORA_RANK, VERA_RANK, BLOCK_DIAG_RANK, PISSA_RANK, DORA_RANK, PROLORA_RANK, ADALORA_RANK, MOS_RANK
    )
    
    experiment_data["parameter_counts"] = param_counts
    
    print(param_counts)
    
    print("\n===== Parameter Counts =====")
    print(f"Base Model Total: {param_counts['base_model']}")
    print(f"First Layer Weights: {param_counts['layer_weights']}")
    print(f"First Layer Total: {param_counts['layer_total']}")
    print(f"LoRA (Rank {LORA_RANK}): {param_counts['lora']}")
    print(f"VeRA (Rank {VERA_RANK}): {param_counts['vera']}")
    print(f"Block Diagonal (Rank {BLOCK_DIAG_RANK}): {param_counts['block_diagonal']}")
    print(f"PiSSA (Rank {PISSA_RANK}): {param_counts['pissa']}")
    print(f"DoRA (Rank {DORA_RANK}): {param_counts['dora']}")
    print(f"ProLoRA (Rank {PROLORA_RANK}): {param_counts['prolora']}")
    print(f"AdaLoRA (Rank {ADALORA_RANK}): {param_counts['adalora']}")
    print(f"MoS (Rank {MOS_RANK}): {param_counts['mos']}")

    # Train base model on original function
    print("\n===== Phase 1: Training Base Model on Original Function =====")
    base_model = BaseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    trained_base_model, base_history, base_params = train_base_model(
        base_model, x_train_base, y_train_base, x_valid_base, y_valid_base, 
        BASE_EPOCHS, BASE_LR, EVAL_INTERVAL
    )

    experiment_data["base_model_training"] = {
        "total_params": base_params,
        "history": {
            "epochs": base_history["epochs"],
            "train_loss": [float(loss) for loss in base_history["train_loss"]],
            "valid_loss": [float(loss) for loss in base_history["valid_loss"]]
        }
    }

    # Phase 2: Adaptation to the modified function
    print("\n===== Phase 2: Adapting to Modified Function =====")
    results = collections.defaultdict(list)
    experiment_data["results"] = {}

    # 1. Full Fine-tuning (all parameters)
    print("\n----- Full Fine-tuning -----")
    full_finetune_model = BaseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    # Load the base model weights
    full_finetune_model.load_state_dict(trained_base_model.state_dict())
    full_finetune_model.adapter_name = "FullFineTune"
    # Train all parameters
    full_finetune_history, full_ft_params, full_ft_train_loss, full_ft_valid_loss, full_ft_detailed = train_adapter(
        full_finetune_model, x_train_adapt, y_train_adapt, x_valid_adapt, y_valid_adapt,
        ADAPT_EPOCHS, ADAPT_LR, EVAL_INTERVAL
    )
    results['full_finetune'].append({
        'params': full_ft_params,
        'train_loss': full_ft_train_loss,
        'valid_loss': full_ft_valid_loss,
        'history': full_finetune_history,
        'detailed_history': full_ft_detailed
    })
    
    experiment_data["results"]["full_finetune"] = {
        "params": full_ft_params,
        "final_train_loss": float(full_ft_train_loss),
        "final_valid_loss": float(full_ft_valid_loss),
        "detailed_history": {
            "step": full_ft_detailed["step"],
            "epoch": full_ft_detailed["epoch"],
            "train_loss": [float(loss) for loss in full_ft_detailed["train_loss"]],
            "valid_loss": [float(loss) for loss in full_ft_detailed["valid_loss"]]
        }
    }

    # 2. LoRA Adaptation
    print("----- LoRA Adaptation -----")
    lora_model = BaseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    lora_model.load_state_dict(trained_base_model.state_dict())
    lora_model.layer1 = LoRAAdapter(lora_model.layer1, LORA_RANK, scale=1.0).to(DEVICE)
    lora_model.adapter_name = f"LoRA (Rank {LORA_RANK})"
    lora_history, lora_params, lora_train_loss, lora_valid_loss, lora_detailed = train_adapter(
        lora_model, x_train_adapt, y_train_adapt, x_valid_adapt, y_valid_adapt,
        ADAPT_EPOCHS, ADAPT_LR, EVAL_INTERVAL
    )
    results['lora'].append({
        'rank': LORA_RANK,
        'params': lora_params,
        'train_loss': lora_train_loss,
        'valid_loss': lora_valid_loss,
        'history': lora_history,
        'detailed_history': lora_detailed
    })
    
    experiment_data["results"]["lora"] = {
        "rank": LORA_RANK,
        "params": lora_params,
        "final_train_loss": float(lora_train_loss),
        "final_valid_loss": float(lora_valid_loss),
        "detailed_history": {
            "step": lora_detailed["step"],
            "epoch": lora_detailed["epoch"],
            "train_loss": [float(loss) for loss in lora_detailed["train_loss"]],
            "valid_loss": [float(loss) for loss in lora_detailed["valid_loss"]]
        }
    }

    # 3. VeRA Adaptation
    print("----- VeRA Adaptation -----")
    vera_model = BaseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    vera_model.load_state_dict(trained_base_model.state_dict())
    vera_model.layer1 = VeRAAdapter(vera_model.layer1, VERA_RANK, d_init_val=0.1).to(DEVICE)
    vera_model.adapter_name = f"VeRA (Rank {VERA_RANK})"
    vera_history, vera_params, vera_train_loss, vera_valid_loss, vera_detailed = train_adapter(
        vera_model, x_train_adapt, y_train_adapt, x_valid_adapt, y_valid_adapt,
        ADAPT_EPOCHS, ADAPT_LR, EVAL_INTERVAL
    )
    results['vera'].append({
        'rank': VERA_RANK,
        'params': vera_params,
        'train_loss': vera_train_loss,
        'valid_loss': vera_valid_loss,
        'history': vera_history,
        'detailed_history': vera_detailed
    })
    
    experiment_data["results"]["vera"] = {
        "rank": VERA_RANK,
        "params": vera_params,
        "final_train_loss": float(vera_train_loss),
        "final_valid_loss": float(vera_valid_loss),
        "detailed_history": {
            "step": vera_detailed["step"],
            "epoch": vera_detailed["epoch"],
            "train_loss": [float(loss) for loss in vera_detailed["train_loss"]],
            "valid_loss": [float(loss) for loss in vera_detailed["valid_loss"]]
        }
    }

    # 4. Block Diagonal Method
    print("----- Block Diagonal Adaptation -----")
    our_model = BaseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    our_model.load_state_dict(trained_base_model.state_dict())
    our_model.layer1 = BlockDiagonalAdapter(our_model.layer1, BLOCK_DIAG_RANK).to(DEVICE)
    our_model.adapter_name = f"BlockDiagonal (Rank {BLOCK_DIAG_RANK})"
    our_history, our_params, our_train_loss, our_valid_loss, our_detailed = train_adapter(
        our_model, x_train_adapt, y_train_adapt, x_valid_adapt, y_valid_adapt,
        ADAPT_EPOCHS, ADAPT_LR, EVAL_INTERVAL
    )
    results['block_diagonal'].append({
        'rank': BLOCK_DIAG_RANK,
        'params': our_params,
        'train_loss': our_train_loss,
        'valid_loss': our_valid_loss,
        'history': our_history,
        'detailed_history': our_detailed
    })
    
    experiment_data["results"]["block_diagonal"] = {
        "rank": BLOCK_DIAG_RANK,
        "params": our_params,
        "final_train_loss": float(our_train_loss),
        "final_valid_loss": float(our_valid_loss),
        "detailed_history": {
            "step": our_detailed["step"],
            "epoch": our_detailed["epoch"],
            "train_loss": [float(loss) for loss in our_detailed["train_loss"]],
            "valid_loss": [float(loss) for loss in our_detailed["valid_loss"]]
        }
    }
    
    # 5. PiSSA Adaptation
    print("----- PiSSA Adaptation -----")
    pissa_model = BaseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    pissa_model.load_state_dict(trained_base_model.state_dict())
    pissa_model.layer1 = PiSSAAdapter(pissa_model.layer1, PISSA_RANK).to(DEVICE)
    pissa_history, pissa_params, pissa_train_loss, pissa_valid_loss, pissa_detailed = train_adapter(
        pissa_model, x_train_adapt, y_train_adapt, x_valid_adapt, y_valid_adapt,
        ADAPT_EPOCHS, ADAPT_LR, EVAL_INTERVAL
    )
    results['pissa'].append({
        'rank': PISSA_RANK,
        'params': pissa_params,
        'train_loss': pissa_train_loss,
        'valid_loss': pissa_valid_loss,
        'history': pissa_history,
        'detailed_history': pissa_detailed
    })
    
    experiment_data["results"]["pissa"] = {
        "rank": PISSA_RANK,
        "params": pissa_params,
        "final_train_loss": float(pissa_train_loss),
        "final_valid_loss": float(pissa_valid_loss),
        "detailed_history": {
            "step": pissa_detailed["step"],
            "epoch": pissa_detailed["epoch"],
            "train_loss": [float(loss) for loss in pissa_detailed["train_loss"]],
            "valid_loss": [float(loss) for loss in pissa_detailed["valid_loss"]]
        }
    }
    
    # 6. DoRA Adaptation
    print("----- DoRA Adaptation -----")
    dora_model = BaseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    dora_model.load_state_dict(trained_base_model.state_dict())
    dora_model.layer1 = DoRAAdapter(dora_model.layer1, DORA_RANK, lora_alpha=1.0).to(DEVICE)
    dora_history, dora_params, dora_train_loss, dora_valid_loss, dora_detailed = train_adapter(
        dora_model, x_train_adapt, y_train_adapt, x_valid_adapt, y_valid_adapt,
        ADAPT_EPOCHS, ADAPT_LR, EVAL_INTERVAL
    )
    results['dora'].append({
        'rank': DORA_RANK,
        'params': dora_params,
        'train_loss': dora_train_loss,
        'valid_loss': dora_valid_loss,
        'history': dora_history,
        'detailed_history': dora_detailed
    })
    
    experiment_data["results"]["dora"] = {
        "rank": DORA_RANK,
        "params": dora_params,
        "final_train_loss": float(dora_train_loss),
        "final_valid_loss": float(dora_valid_loss),
        "detailed_history": {
            "step": dora_detailed["step"],
            "epoch": dora_detailed["epoch"],
            "train_loss": [float(loss) for loss in dora_detailed["train_loss"]],
            "valid_loss": [float(loss) for loss in dora_detailed["valid_loss"]]
        }
    }
    
    # 7. ProLoRA Adaptation
    print("----- ProLoRA Adaptation -----")
    prolora_model = BaseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    prolora_model.load_state_dict(trained_base_model.state_dict())
    # Using default values for unshared_rank=1, sharing_ratio_m_A=2, sharing_ratio_n_B=2
    prolora_model.layer1 = ProLoRAAdapter(prolora_model.layer1, PROLORA_RANK).to(DEVICE)
    prolora_history, prolora_params, prolora_train_loss, prolora_valid_loss, prolora_detailed = train_adapter(
        prolora_model, x_train_adapt, y_train_adapt, x_valid_adapt, y_valid_adapt,
        ADAPT_EPOCHS, ADAPT_LR, EVAL_INTERVAL
    )
    results['prolora'].append({
        'rank': PROLORA_RANK,
        'params': prolora_params,
        'train_loss': prolora_train_loss,
        'valid_loss': prolora_valid_loss,
        'history': prolora_history,
        'detailed_history': prolora_detailed
    })
    
    experiment_data["results"]["prolora"] = {
        "rank": PROLORA_RANK,
        "params": prolora_params,
        "final_train_loss": float(prolora_train_loss),
        "final_valid_loss": float(prolora_valid_loss),
        "detailed_history": {
            "step": prolora_detailed["step"],
            "epoch": prolora_detailed["epoch"],
            "train_loss": [float(loss) for loss in prolora_detailed["train_loss"]],
            "valid_loss": [float(loss) for loss in prolora_detailed["valid_loss"]]
        }
    }
    
    # 8. AdaLoRA Adaptation
    print("----- AdaLoRA Adaptation -----")
    adalora_model = BaseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    adalora_model.load_state_dict(trained_base_model.state_dict())
    adalora_model.layer1 = AdaLoRAAdapter(adalora_model.layer1, ADALORA_RANK).to(DEVICE)
    adalora_history, adalora_params, adalora_train_loss, adalora_valid_loss, adalora_detailed = train_adapter(
        adalora_model, x_train_adapt, y_train_adapt, x_valid_adapt, y_valid_adapt,
        ADAPT_EPOCHS, ADAPT_LR, EVAL_INTERVAL
    )
    results['adalora'].append({
        'rank': ADALORA_RANK,
        'params': adalora_params,
        'train_loss': adalora_train_loss,
        'valid_loss': adalora_valid_loss,
        'history': adalora_history,
        'detailed_history': adalora_detailed
    })
    
    experiment_data["results"]["adalora"] = {
        "rank": ADALORA_RANK,
        "params": adalora_params,
        "final_train_loss": float(adalora_train_loss),
        "final_valid_loss": float(adalora_valid_loss),
        "detailed_history": {
            "step": adalora_detailed["step"],
            "epoch": adalora_detailed["epoch"],
            "train_loss": [float(loss) for loss in adalora_detailed["train_loss"]],
            "valid_loss": [float(loss) for loss in adalora_detailed["valid_loss"]]
        }
    }
    
    # 9. MoS Adaptation
    print("----- MoS Adaptation -----")
    mos_model = BaseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    mos_model.load_state_dict(trained_base_model.state_dict())
    mos_model.layer1 = MoSAdapter(mos_model.layer1, MOS_RANK).to(DEVICE)
    mos_history, mos_params, mos_train_loss, mos_valid_loss, mos_detailed = train_adapter(
        mos_model, x_train_adapt, y_train_adapt, x_valid_adapt, y_valid_adapt,
        ADAPT_EPOCHS, ADAPT_LR, EVAL_INTERVAL
    )
    results['mos'].append({
        'rank': MOS_RANK,
        'params': mos_params,
        'train_loss': mos_train_loss,
        'valid_loss': mos_valid_loss,
        'history': mos_history,
        'detailed_history': mos_detailed
    })
    
    experiment_data["results"]["mos"] = {
        "rank": MOS_RANK,
        "params": mos_params,
        "final_train_loss": float(mos_train_loss),
        "final_valid_loss": float(mos_valid_loss),
        "detailed_history": {
            "step": mos_detailed["step"],
            "epoch": mos_detailed["epoch"],
            "train_loss": [float(loss) for loss in mos_detailed["train_loss"]],
            "valid_loss": [float(loss) for loss in mos_detailed["valid_loss"]]
        }
    }

    # --- Output Results ---
    print("\n\n===== Final Performance Summary =====")
    summary = []

    # Add full fine-tuning results
    for method_name, method_data in experiment_data["results"].items():
        if method_name == "full_finetune":
            summary.append({
                'method': "FULL_FINETUNE",
                'rank': "N/A",
                'params': method_data['params'],
                'train_loss': method_data['final_train_loss'],
                'valid_loss': method_data['final_valid_loss'],
                'init_time_ms': method_data.get('init_time_ms', "N/A") # Full FT doesn't have adapter init
            })
        else:
            summary.append({
                'method': method_name.upper(),
                'rank': method_data['rank'],
                'params': method_data['params'],
                'train_loss': method_data['final_train_loss'],
                'valid_loss': method_data['final_valid_loss'],
                'init_time_ms': method_data.get('init_time_ms', "N/A")
            })
    
    # Sort by parameter count then by initialization time
    summary.sort(key=lambda x: (x['params'], x['init_time_ms'] if isinstance(x['init_time_ms'], (int, float)) else float('inf')))
    print("\nMethod, Rank, Params, TrainLoss, ValidLoss, InitTime (ms)")
    for item in summary:
        init_time_str = f"{item['init_time_ms']:.4f}" if isinstance(item['init_time_ms'], (int, float)) else item['init_time_ms']
        print(f"{item['method']}, Rank={item['rank']}, Params={item['params']}, TrainLoss={item['train_loss']:.6f}, ValidLoss={item['valid_loss']:.6f}, InitTime={init_time_str}")

    # Print detailed training curve data for selected epochs
    print("\n\n===== Training Curve Data =====")
    output_epochs = [1, 10, 100, ADAPT_EPOCHS]

    for method, method_results in results.items():
        for run_data in method_results:
            rank = run_data.get('rank', 'N/A')
            params = run_data['params']
            history = run_data['history']
            
            print(f"\n--- Method: {method.upper()}, Rank: {rank}, Params: {params} ---")
            print("Epoch, TrainLoss")
            
            epochs = history['epochs']
            train_losses = history['train_loss']
            
            for i, epoch in enumerate(epochs):
                if epoch in output_epochs or epoch == epochs[-1]:
                    print(f"{epoch}, {train_losses[i]:.8f}")
    
    # Save the experiment data to a JSON file
    with open(result_json_path, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"\nExperiment data saved to: {result_json_path}")
    print(f"\nExperiment completed on device: {DEVICE}")

# --- Updated Benchmarking Function ---
def benchmark_adapter_initialization(
    adapter_class,
    original_layer_supplier,
    rank_key_in_params: str,
    num_runs=10,
    warmup_runs=2, # Added warmup_runs
    device=torch.device("cpu"), # Pass device explicitly
    **kwargs_for_adapter
):
    """Benchmarks the initialization time of a given adapter with CUDA sync and warmup."""
    total_time_ns = 0
    adapter_constructor_args = kwargs_for_adapter.copy()

    if rank_key_in_params not in adapter_constructor_args:
        raise ValueError(
            f"Rank key '{rank_key_in_params}' not found in adapter_specific_kwargs: {adapter_constructor_args}"
        )

    rank_value = adapter_constructor_args.pop(rank_key_in_params)

    # Warm-up runs
    for _ in range(warmup_runs):
        fresh_original_layer_warmup = original_layer_supplier().to(device)
        # Ensure the original layer is fully on device before adapter uses it
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        adapter_instance_warmup = adapter_class(
            fresh_original_layer_warmup, rank_value, **adapter_constructor_args
        )
        if hasattr(adapter_instance_warmup, 'to') and isinstance(adapter_instance_warmup, nn.Module):
            adapter_instance_warmup.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize(device) # Synchronize after adapter creation and move
        del fresh_original_layer_warmup, adapter_instance_warmup
        if device.type == 'cuda':
            torch.cuda.empty_cache() # Good practice during benchmarking loops


    # Actual benchmark runs
    for _ in range(num_runs):
        fresh_original_layer = original_layer_supplier().to(device)
        # Ensure the original layer is fully on device before adapter uses it
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        start_time = time.perf_counter_ns()

        adapter_instance = adapter_class(
            fresh_original_layer, rank_value, **adapter_constructor_args
        )

        if hasattr(adapter_instance, 'to') and isinstance(adapter_instance, nn.Module):
            adapter_instance.to(device) # Move adapter parameters to device

        # CRITICAL: Synchronize CUDA device to ensure all operations are complete
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        end_time = time.perf_counter_ns()
        total_time_ns += (end_time - start_time)
        del fresh_original_layer, adapter_instance # Clean up
        if device.type == 'cuda':
            torch.cuda.empty_cache()


    avg_time_ms = (total_time_ns / num_runs) / 1_000_000  # Convert ns to ms
    return avg_time_ms

# --- Helper to get parameter count for an adapter config ---
def get_adapter_param_count(adapter_class, original_layer_supplier, rank_key_in_params: str, **kwargs_for_adapter):
    """Instantiates an adapter on CPU and returns its parameter count."""
    temp_kwargs = kwargs_for_adapter.copy()
    rank_value = temp_kwargs.pop(rank_key_in_params)
    
    # Use a CPU layer for param count to avoid CUDA interactions if not needed
    cpu_layer = original_layer_supplier() # Assuming supplier gives CPU layer or it's moved by adapter
    # Adapters might move the layer to their device, so ensure this is okay or force CPU.
    # For safety, let's assume original_layer_supplier provides a CPU layer initially.
    # If not, cpu_layer = original_layer_supplier().to(torch.device("cpu"))

    try:
        # Temporarily move to CPU for param count if original layer might be on GPU
        # This is to ensure num_adapter_parameters() doesn't trigger GPU operations unnecessarily
        # if the adapter itself doesn't explicitly handle device for param counting.
        temp_original_layer_cpu = original_layer_supplier().to(torch.device("cpu"))

        adapter_instance = adapter_class(temp_original_layer_cpu, rank_value, **temp_kwargs)
        
        # Some adapters might not have num_adapter_parameters, handle this.
        if hasattr(adapter_instance, 'num_adapter_parameters') and callable(adapter_instance.num_adapter_parameters):
            return adapter_instance.num_adapter_parameters()
        else:
            # Fallback: count all parameters in the adapter if specific method is missing
            # This might include original layer params if not properly frozen/separated by adapter design
            print(f"Warning: {adapter_class.__name__} missing num_adapter_parameters(). Summing all params.")
            return sum(p.numel() for p in adapter_instance.parameters() if p.requires_grad) # Only trainable by default

    except Exception as e:
        print(f"Error getting param count for {adapter_class.__name__} with rank {rank_value}: {e}")
        return -1 # Indicate error

# --- Helper to Benchmark Adapter Training Time ---
def benchmark_adapter_training_time(
    adapter_class,
    original_layer_supplier,
    rank_key_in_params: str,
    num_train_steps: int,
    batch_size: int,
    input_dim: int, # Dimension of input to the layer the adapter is on
    # Output dim of the layer the adapter is on (e.g., hidden_dim if layer1)
    output_dim_of_adapted_layer: int, 
    num_runs=5, # Fewer runs than init, as training steps are slower
    warmup_runs=1, # Fewer warmup runs, or more steps per warmup
    warmup_steps_per_run=5, # Number of training steps during each warmup run
    device=torch.device("cpu"),
    **kwargs_for_adapter
):
    """Benchmarks the training time (fwd, loss, bwd, step) of a given adapter."""
    total_time_ns = 0
    adapter_constructor_args = kwargs_for_adapter.copy()

    if rank_key_in_params not in adapter_constructor_args:
        raise ValueError(
            f"Rank key '{rank_key_in_params}' not found in adapter_specific_kwargs: {adapter_constructor_args}"
        )
    rank_value = adapter_constructor_args.pop(rank_key_in_params)

    criterion = nn.MSELoss()

    # Warm-up runs
    for i_warmup in range(warmup_runs):
        # print(f"Warmup run {i_warmup + 1}/{warmup_runs}...")
        fresh_original_layer_warmup = original_layer_supplier().to(device)
        adapter_instance_warmup = adapter_class(
            fresh_original_layer_warmup, rank_value, **adapter_constructor_args
        )
        if hasattr(adapter_instance_warmup, 'to') and isinstance(adapter_instance_warmup, nn.Module):
            adapter_instance_warmup.to(device)
        
        # Ensure all parameters are on the correct device
        # Forcing parameters to device again after potential internal moves
        for param in adapter_instance_warmup.parameters():
            param.to(device)
        
        trainable_params_warmup = [p for p in adapter_instance_warmup.parameters() if p.requires_grad]
        if not trainable_params_warmup:
            # print(f"Skipping warmup for {adapter_class.__name__} rank {rank_value}: No trainable parameters.")
            del fresh_original_layer_warmup, adapter_instance_warmup
            if device.type == 'cuda': torch.cuda.empty_cache()
            continue # Skip if no trainable params

        optimizer_warmup = optim.Adam(trainable_params_warmup, lr=1e-3)

        for _ in range(warmup_steps_per_run): # Perform a few training steps for warmup
            dummy_x = torch.randn(batch_size, input_dim, device=device)
            # The target for the loss should match the output shape of the adapted layer
            dummy_y = torch.randn(batch_size, output_dim_of_adapted_layer, device=device)

            if device.type == 'cuda': torch.cuda.synchronize(device)
            
            optimizer_warmup.zero_grad()
            output = adapter_instance_warmup(dummy_x)
            loss = criterion(output, dummy_y)
            loss.backward()
            optimizer_warmup.step()
            
            if device.type == 'cuda': torch.cuda.synchronize(device)
        
        del fresh_original_layer_warmup, adapter_instance_warmup, optimizer_warmup, trainable_params_warmup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Actual benchmark runs
    for i_run in range(num_runs):
        # print(f"Benchmark run {i_run + 1}/{num_runs}...")
        fresh_original_layer = original_layer_supplier().to(device)
        adapter_instance = adapter_class(
            fresh_original_layer, rank_value, **adapter_constructor_args
        )
        if hasattr(adapter_instance, 'to') and isinstance(adapter_instance, nn.Module):
            adapter_instance.to(device)

        # Ensure all parameters are on the correct device
        for param in adapter_instance.parameters():
            param.to(device)

        trainable_params = [p for p in adapter_instance.parameters() if p.requires_grad]
        if not trainable_params:
            # print(f"Skipping training benchmark for {adapter_class.__name__} rank {rank_value}: No trainable parameters.")
            # This adapter config likely has no trainable parameters.
            # Return 0 or a special value, or raise error if this is unexpected.
            # For now, if an adapter has zero trainable params, its training time is effectively zero.
            del fresh_original_layer, adapter_instance
            if device.type == 'cuda': torch.cuda.empty_cache()
            return 0.0


        optimizer = optim.Adam(trainable_params, lr=1e-3)
        
        run_time_ns = 0
        for step in range(num_train_steps):
            dummy_x = torch.randn(batch_size, input_dim, device=device)
            dummy_y = torch.randn(batch_size, output_dim_of_adapted_layer, device=device)

            if device.type == 'cuda':
                torch.cuda.synchronize(device) # Sync before starting timer for the step
            
            start_step_time = time.perf_counter_ns()

            optimizer.zero_grad()
            output = adapter_instance(dummy_x) # Adapter's forward includes original layer
            loss = criterion(output, dummy_y)
            loss.backward()
            optimizer.step()

            if device.type == 'cuda':
                torch.cuda.synchronize(device) # Sync after operations for the step
            
            end_step_time = time.perf_counter_ns()
            run_time_ns += (end_step_time - start_step_time)
        
        total_time_ns += (run_time_ns / num_train_steps) # Average time per step for this run

        del fresh_original_layer, adapter_instance, optimizer, trainable_params
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    avg_time_ms_per_step = (total_time_ns / num_runs) / 1_000_000  # Convert ns to ms
    return avg_time_ms_per_step

# --- Updated Experiment Function ---
def run_initialization_benchmark_experiment(get_fresh_linear_layer_supplier):
    """Runs the initialization benchmark with CUDA sync and warmup, varying ranks."""
    print(f"\n===== Running Initialization Benchmark vs. Parameter Count on device: {DEVICE} =====")

    num_benchmark_runs = 20 
    num_warmup_runs = 5    

    # Define ranks to test. Filtered per adapter if needed (e.g., BlockDiagonal).
    ranks_to_test = [1, 2, 4, 8, 16, 32, 64]
    if INPUT_DIM < 64: # Adjust if INPUT_DIM is smaller
        ranks_to_test = [r for r in ranks_to_test if r <= INPUT_DIM]

    # Base adapter configurations (rank will be overridden in the loop)
    base_adapter_configs = {
        "LoRA": {"adapter_class": LoRAAdapter, "rank_key": "rank", 
                 "params": {"scale": 1.0}},
        "VeRA": {"adapter_class": VeRAAdapter, "rank_key": "rank", 
                 "params": {"d_init_val": 0.1}},
        "BlockDiagonal": {"adapter_class": BlockDiagonalAdapter, "rank_key": "rank", 
                          "params": {}},
        "PiSSA": {"adapter_class": PiSSAAdapter, "rank_key": "rank", 
                  "params": {}},
        "DoRA": {"adapter_class": DoRAAdapter, "rank_key": "rank", 
                 "params": {"lora_alpha": 1.0}},
        "ProLoRA": {"adapter_class": ProLoRAAdapter, "rank_key": "rank", 
                    "params": { # unshared_rank_u and others will be set based on current_rank
                               "sharing_ratio_m_A": None, "sharing_ratio_n_B": None,
                               "base_stride_sA":1, "base_stride_sB":1}},
        "AdaLoRA": {"adapter_class": AdaLoRAAdapter, "rank_key": "initial_rank",
                    "params": {}},
        "MoS": {"adapter_class": MoSAdapter, "rank_key": "rank_per_layer", 
                "params": { # shard_dim_ratio will be set based on current_rank
                          }},
    }

    all_results_data = []
    print(f"\nBenchmarking initialization time vs. params for each adapter ({num_benchmark_runs} runs, {num_warmup_runs} warmup runs each)...")

    for name, base_config in base_adapter_configs.items():
        adapter_class = base_config["adapter_class"]
        rank_key = base_config["rank_key"]
        print(f"\n--- Benchmarking Adapter: {name} ---")

        current_ranks_for_adapter = ranks_to_test
        if name == "BlockDiagonal":
            # Filter ranks for BlockDiagonal: must be divisors of INPUT_DIM
            current_ranks_for_adapter = [r for r in ranks_to_test if INPUT_DIM % r == 0]
            if not current_ranks_for_adapter:
                print(f"Skipping {name} as no valid ranks found (INPUT_DIM={INPUT_DIM}, tested ranks up to 64).")
                continue
        
        for current_rank_val in current_ranks_for_adapter:
            current_adapter_specific_params = base_config["params"].copy()
            current_adapter_specific_params[rank_key] = current_rank_val

            # Adjust dependent parameters for ProLoRA and MoS based on current_rank_val
            if name == "ProLoRA":
                current_adapter_specific_params["unshared_rank_u"] = max(1, current_rank_val // 4)
            elif name == "MoS":
                current_adapter_specific_params["shard_dim_ratio"] = max(1, current_rank_val // 2)
                # Ensure shard_dim_ratio is at least 1 and rank is divisible if adapter requires
                if current_adapter_specific_params["shard_dim_ratio"] == 0: current_adapter_specific_params["shard_dim_ratio"] =1
                if current_rank_val % current_adapter_specific_params["shard_dim_ratio"] != 0:
                     # Find the largest divisor of current_rank_val for shard_dim_ratio or default to 1 or 2
                    if current_rank_val > 1 and current_rank_val % 2 == 0 : current_adapter_specific_params["shard_dim_ratio"] = 2
                    else: current_adapter_specific_params["shard_dim_ratio"] = 1
                    print(f"Adjusted MoS shard_dim_ratio to {current_adapter_specific_params['shard_dim_ratio']} for rank {current_rank_val}")

            # 1. Get parameter count
            # Need to pass all expected params to get_adapter_param_count, including rank_key
            num_params = get_adapter_param_count(
                adapter_class,
                get_fresh_linear_layer_supplier, # Supplier for CPU layer for param count
                rank_key_in_params=rank_key,
                **current_adapter_specific_params
            )

            if num_params == -1: # Error occurred during param count
                print(f"Skipping benchmark for {name} at rank {current_rank_val} due to param count error.")
                all_results_data.append({
                    "method": name, 
                    "rank_config": current_rank_val, 
                    "params_for_rank_key": current_adapter_specific_params.get(rank_key),
                    "full_config_params": current_adapter_specific_params,
                    "num_params": "Error", 
                    "init_time_ms": "Error (param count failed)"
                })
                continue
            
            print(f"Benchmarking {name} (Rank for key '{rank_key}': {current_rank_val}, Calculated Params: {num_params})...")
            
            # 2. Benchmark initialization time
            try:
                # The kwargs for benchmark_adapter_initialization must also include the rank_key field
                init_time_ms = benchmark_adapter_initialization(
                    adapter_class,
                    get_fresh_linear_layer_supplier, # Supplier for potentially GPU layer for benchmark
                    rank_key_in_params=rank_key,
                    num_runs=num_benchmark_runs,
                    warmup_runs=num_warmup_runs,
                    device=DEVICE,          
                    **current_adapter_specific_params 
                )
                all_results_data.append({
                    "method": name, 
                    "rank_config": current_rank_val, # This is the value from ranks_to_test
                    "params_for_rank_key": current_adapter_specific_params.get(rank_key), # Actual rank value used
                    "full_config_params": current_adapter_specific_params.copy(), # Store full config for this run
                    "num_params": num_params, 
                    "init_time_ms": init_time_ms
                })
                print(f"{name} (Rank '{rank_key}'={current_rank_val}, Params={num_params}) - Avg Init Time: {init_time_ms:.4f} ms")
            except Exception as e:
                print(f"Error benchmarking {name} at rank {current_rank_val}: {e}")
                all_results_data.append({
                    "method": name, 
                    "rank_config": current_rank_val, 
                    "params_for_rank_key": current_adapter_specific_params.get(rank_key),
                    "full_config_params": current_adapter_specific_params,
                    "num_params": num_params, # Params might be available even if init fails
                    "init_time_ms": f"Error: {str(e)}"
                })

    print("\n\n===== Overall Initialization Benchmark vs. Parameter Count Summary =====")
    # Sort by method, then by num_params for consistent output
    # Handle cases where num_params might be string "Error"
    def sort_key(x):
        num_p = x['num_params']
        if isinstance(num_p, str):
            return (x['method'], float('inf')) # Put errors at the end for each method
        return (x['method'], num_p)

    all_results_data.sort(key=sort_key)
    
    print("Method, Rank Key, Rank Value, Num Params, Avg Init Time (ms)")
    for item in all_results_data:
        rank_val_str = str(item['params_for_rank_key'])
        # Find the actual rank key used for this method for clarity in print
        actual_rank_key_used = "unknown_rank_key"
        for m_name, m_cfg in base_adapter_configs.items():
            if m_name == item['method']:
                actual_rank_key_used = m_cfg["rank_key"]
                break
        print(f"{item['method']}, {actual_rank_key_used}, {rank_val_str}, {item['num_params']}, {item['init_time_ms'] if isinstance(item['init_time_ms'], (float, int)) else str(item['init_time_ms'])}")

    results_dir = "experiment_results_init_vs_params"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_json_path = os.path.join(results_dir, f"init_vs_params_benchmark_{timestamp}.json")
    
    output_data = {
        "experiment_description": "Adapter Initialization Speed vs. Parameter Count (CUDA Synchronized & Warmed Up)",
        "config": {
            "device": str(DEVICE),
            "input_dim_for_test_layer": INPUT_DIM,
            "hidden_dim_for_test_layer": HIDDEN_DIM,
            "num_benchmark_runs_per_config": num_benchmark_runs,
            "num_warmup_runs_per_config": num_warmup_runs,
            "ranks_tested_range": ranks_to_test,
            "seed": SEED,
        },
        "benchmark_results": all_results_data 
    }
    with open(result_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nInitialization benchmark data (vs. params) saved to: {result_json_path}")
    print(f"\nInitialization benchmark experiment completed on device: {DEVICE}")

# --- Experiment Function for Training Time Benchmark ---
def run_training_time_benchmark_experiment(get_fresh_linear_layer_supplier):
    """Runs the training time benchmark, varying ranks, and collects results."""
    print(f"\n===== Running Training Time Benchmark vs. Parameter Count on device: {DEVICE} =====")

    # Experiment Parameters
    num_benchmark_runs = 10  # Number of times to repeat the benchmark for each config
    num_warmup_runs = 2     # Number of warmup cycles before actual benchmark
    warmup_steps_per_run_cycle = 10 # Training steps in each warmup cycle
    
    num_train_steps_benchmark = 20 # Number of training steps for the actual benchmark measurement
    batch_size_benchmark = 32       # Batch size for dummy data

    # Define ranks to test.
    ranks_to_test = [1, 2, 4, 8, 16, 32, 64]
    if INPUT_DIM < 64: # Adjust if INPUT_DIM is smaller
        ranks_to_test = [r for r in ranks_to_test if r <= INPUT_DIM]
    if HIDDEN_DIM < 64 and any(r > HIDDEN_DIM for r in ranks_to_test):
        # Some ranks might be too large for out_features if it's HIDDEN_DIM
        # This depends on adapter implementation; for now, we allow it
        # as adapters like LoRA have rank * in_features + out_features * rank.
        pass

    base_adapter_configs = {
        "LoRA": {"adapter_class": LoRAAdapter, "rank_key": "rank", "params": {"scale": 1.0}},
        "VeRA": {"adapter_class": VeRAAdapter, "rank_key": "rank", "params": {"d_init_val": 0.1}},
        "BlockDiagonal": {"adapter_class": BlockDiagonalAdapter, "rank_key": "rank", "params": {}},
        "PiSSA": {"adapter_class": PiSSAAdapter, "rank_key": "rank", "params": {}},
        "DoRA": {"adapter_class": DoRAAdapter, "rank_key": "rank", "params": {"lora_alpha": 1.0}},
        "ProLoRA": {"adapter_class": ProLoRAAdapter, "rank_key": "rank", 
                    "params": {"sharing_ratio_m_A": None, "sharing_ratio_n_B": None, "base_stride_sA":1, "base_stride_sB":1}},
        "AdaLoRA": {"adapter_class": AdaLoRAAdapter, "rank_key": "initial_rank", "params": {}},
        "MoS": {"adapter_class": MoSAdapter, "rank_key": "rank_per_layer", "params": {}},
    }

    all_training_results_data = []
    print(f"\nBenchmarking training time vs. params for each adapter ({num_benchmark_runs} runs, {num_warmup_runs} warmup runs with {warmup_steps_per_run_cycle} steps each, benchmark: {num_train_steps_benchmark} steps)...")

    # The layer being adapted is nn.Linear(INPUT_DIM, HIDDEN_DIM)
    # So, input_dim for training benchmark is INPUT_DIM
    # And output_dim_of_adapted_layer for training benchmark is HIDDEN_DIM
    training_input_dim = INPUT_DIM
    training_output_dim_adapted = HIDDEN_DIM

    for name, base_config in base_adapter_configs.items():
        adapter_class = base_config["adapter_class"]
        rank_key = base_config["rank_key"]
        print(f"\n--- Benchmarking Training Time for Adapter: {name} ---")

        current_ranks_for_adapter = ranks_to_test
        if name == "BlockDiagonal":
            current_ranks_for_adapter = [r for r in ranks_to_test if INPUT_DIM % r == 0]
            if not current_ranks_for_adapter:
                print(f"Skipping {name} for training benchmark as no valid ranks found.")
                continue
        
        for current_rank_val in current_ranks_for_adapter:
            current_adapter_specific_params = base_config["params"].copy()
            current_adapter_specific_params[rank_key] = current_rank_val

            if name == "ProLoRA":
                current_adapter_specific_params["unshared_rank_u"] = max(1, current_rank_val // 4)
            elif name == "MoS":
                current_adapter_specific_params["shard_dim_ratio"] = max(1, current_rank_val // 2)
                if current_adapter_specific_params["shard_dim_ratio"] == 0: current_adapter_specific_params["shard_dim_ratio"] = 1
                if current_rank_val % current_adapter_specific_params["shard_dim_ratio"] != 0:
                    if current_rank_val > 1 and current_rank_val % 2 == 0 : current_adapter_specific_params["shard_dim_ratio"] = 2
                    else: current_adapter_specific_params["shard_dim_ratio"] = 1

            # 1. Get parameter count (using CPU to avoid GPU mem issues if many configs)
            num_params = get_adapter_param_count(
                adapter_class,
                get_fresh_linear_layer_supplier, # For CPU layer
                rank_key_in_params=rank_key,
                **current_adapter_specific_params
            )

            if num_params == -1:
                print(f"Skipping training benchmark for {name} at rank {current_rank_val} due to param count error.")
                all_training_results_data.append({
                    "method": name, "rank_config": current_rank_val,
                    "params_for_rank_key": current_adapter_specific_params.get(rank_key),
                    "full_config_params": current_adapter_specific_params,
                    "num_params": "Error", "avg_train_time_ms_per_step": "Error (param count failed)"
                })
                continue
            
            print(f"Benchmarking Training: {name} (Rank '{rank_key}'={current_rank_val}, Params: {num_params})...")

            # 2. Benchmark training time
            try:
                avg_train_time_ms = benchmark_adapter_training_time(
                    adapter_class,
                    get_fresh_linear_layer_supplier, # For layer on target DEVICE
                    rank_key_in_params=rank_key,
                    num_train_steps=num_train_steps_benchmark,
                    batch_size=batch_size_benchmark,
                    input_dim=training_input_dim,
                    output_dim_of_adapted_layer=training_output_dim_adapted,
                    num_runs=num_benchmark_runs,
                    warmup_runs=num_warmup_runs,
                    warmup_steps_per_run=warmup_steps_per_run_cycle,
                    device=DEVICE,
                    **current_adapter_specific_params
                )
                all_training_results_data.append({
                    "method": name, "rank_config": current_rank_val,
                    "params_for_rank_key": current_adapter_specific_params.get(rank_key),
                    "full_config_params": current_adapter_specific_params.copy(),
                    "num_params": num_params, "avg_train_time_ms_per_step": avg_train_time_ms
                })
                print(f"{name} (Rank '{rank_key}'={current_rank_val}, Params={num_params}) - Avg Train Time/Step: {avg_train_time_ms:.4f} ms")
            except Exception as e:
                print(f"Error benchmarking training for {name} at rank {current_rank_val}: {e}")
                all_training_results_data.append({
                    "method": name, "rank_config": current_rank_val,
                    "params_for_rank_key": current_adapter_specific_params.get(rank_key),
                    "full_config_params": current_adapter_specific_params,
                    "num_params": num_params, "avg_train_time_ms_per_step": f"Error: {str(e)}"
                })

    print("\n\n===== Overall Training Time Benchmark vs. Parameter Count Summary =====")
    def sort_key_train(x):
        num_p = x['num_params']
        return (x['method'], float('inf') if isinstance(num_p, str) else num_p)
    all_training_results_data.sort(key=sort_key_train)

    print("Method, Rank Key, Rank Value, Num Params, Avg Train Time/Step (ms)")
    for item in all_training_results_data:
        rank_val_str = str(item['params_for_rank_key'])
        actual_rank_key_used = next((m_cfg["rank_key"] for m_name, m_cfg in base_adapter_configs.items() if m_name == item['method']), "unknown_rank_key")
        time_val = item['avg_train_time_ms_per_step']
        time_str = f"{time_val:.4f}" if isinstance(time_val, (float, int)) else str(time_val)
        print(f"{item['method']}, {actual_rank_key_used}, {rank_val_str}, {item['num_params']}, {time_str}")

    results_dir = "experiment_results_training_time_vs_params"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_json_path = os.path.join(results_dir, f"training_time_vs_params_benchmark_{timestamp}.json")
    
    output_data = {
        "experiment_description": "Adapter Training Time per Step vs. Parameter Count (CUDA Synchronized & Warmed Up)",
        "config": {
            "device": str(DEVICE),
            "input_dim_for_test_layer": INPUT_DIM,       # Input dim of the nn.Linear layer being adapted
            "output_dim_for_test_layer": HIDDEN_DIM,     # Output dim of the nn.Linear layer being adapted
            "num_benchmark_runs_per_config": num_benchmark_runs,
            "num_warmup_runs_per_config": num_warmup_runs,
            "warmup_steps_per_run_cycle": warmup_steps_per_run_cycle,
            "num_train_steps_for_benchmark": num_train_steps_benchmark,
            "batch_size_for_benchmark": batch_size_benchmark,
            "ranks_tested_range": ranks_to_test,
            "seed": SEED,
        },
        "benchmark_results": all_training_results_data 
    }
    with open(result_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nTraining time benchmark data (vs. params) saved to: {result_json_path}")
    print(f"\nTraining time benchmark experiment completed on device: {DEVICE}")

# --- Main Execution ---
if __name__ == "__main__":
    # Helper function to supply a fresh linear layer for benchmarking
    def get_fresh_linear_layer():
        # The linear layer maps from INPUT_DIM to HIDDEN_DIM as per original BaseModel
        layer = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        return layer

    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(DEVICE)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")

    # --- Choose which benchmarks to run ---
    # Run the initialization speed benchmark:
    # run_initialization_benchmark_experiment(get_fresh_linear_layer)
    
    # Run the training time benchmark:
    run_training_time_benchmark_experiment(get_fresh_linear_layer)
    
    # Run the original full experiment (training quality):
    # run_experiment()