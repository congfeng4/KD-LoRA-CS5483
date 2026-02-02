import math
import warnings
from typing import Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer

from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose


def generate_mrlora_ranks(highest_rank):
    """Generate MrLoRA ranks list from highest_rank down to 1 by halving."""
    ranks = []
    r = highest_rank
    while r >= 1:
        ranks.append(r)
        r = r // 2

    return ranks


class MrLoraLayer(nn.Module, LoraLayer):
    def __init__(self, in_features, out_features, total_rank, lora_alpha, lora_dropout,
                 init_type: Literal['standard', 'olora']='standard',
                 use_rslora=True,
                 **kwargs):
        nn.Module.__init__(self)
        self.init_type = init_type
        LoraLayer.__init__(self, base_layer=kwargs.get("base_layer"))
        assert total_rank % 2 == 0, total_rank
        self.highest_rank = total_rank // 2
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora
        self.ranks_int = generate_mrlora_ranks(self.highest_rank)
        self.ranks_str = [str(r) for r in self.ranks_int]

        # 1. Pre-compute scaling factors to avoid math in the forward pass
        if use_rslora:
            # RS-LoRA: alpha / sqrt(r)
            scalings = [lora_alpha / math.sqrt(r) for r in self.ranks_int]
        else:
            # Standard: alpha / r (or alpha / max_r as per your original logic)
            # Standard LoRA usually uses alpha / r. 
            scalings = [lora_alpha / r for r in self.ranks_int]
        
        # Register as buffer so it moves with the model to GPU
        self.register_buffer("scaling_factors", torch.tensor(scalings, dtype=torch.float32))

        # Multi-rank components
        self.lora_A = nn.ModuleDict({r_str: nn.Linear(in_features, r_int, bias=False) 
                                     for r_str, r_int in zip(self.ranks_str, self.ranks_int)})
        self.lora_B = nn.ModuleDict({r_str: nn.Linear(r_int, out_features, bias=False) 
                                     for r_str, r_int in zip(self.ranks_str, self.ranks_int)})
        
        # Learnable coefficients (initialized to 1.0 or small normal)
        # Using ones_ helps maintain the initial magnitude of the pre-computed scalings
        self.alphas = nn.Parameter(torch.ones(len(self.ranks_int)))

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.reset_mr_parameters()

    def reset_mr_parameters(self):
        if self.init_type == 'standard':
            self.reset_mr_parameters_standard()
        elif self.init_type == 'olora':
            self.reset_mr_parameters_olora()

    def reset_mr_parameters_olora(self):
        """
        实现基于 OLoRA 思想的 SVD 能量切片初始化：
        1. 对原始权重 W 进行 SVD 分解。
        2. 按奇异值从高到低，将对应的正交基分配给 Rank 4, 2, 1 等矩阵。
        """
        # 1. 获取基础层的权重数据
        with torch.no_grad():
            # 获取 base_layer 权重 [out_features, in_features]
            weight = self.get_base_layer().weight.data.float()

            # 2. 执行 SVD 分解
            # U: [out_features, K], S: [K], Vh: [K, in_features]
            # 这里 K = min(out_features, in_features)
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

            # 3. 按照你的计划进行“能量切片”分配
            current_idx = 0
            for i, r_str in enumerate(self.ranks_str):
                r_int = self.ranks_int[i]

                # 在 SVD 结果中提取对应的分量
                # 取出第 current_idx 到 current_idx + r_int 个奇异向量
                u_slice = U[:, current_idx: current_idx + r_int]
                s_slice = S[current_idx: current_idx + r_int]
                vh_slice = Vh[current_idx: current_idx + r_int, :]

                # 4. 初始化 A：分配右奇异向量 (正交基)
                # lora_A 的形状通常是 [r, in_features]
                self.lora_A[r_str].weight.data.copy_(vh_slice.to(self.lora_A[r_str].weight.dtype))

                # 5. 初始化 B：分配左奇异向量并结合奇异值
                # 理论上 AB = u_slice @ diag(s_slice) @ vh_slice
                # 为了抵消 forward 中的 scaling_factor，这里需要除以它
                b_init = u_slice @ torch.diag(s_slice)
                b_init = b_init / self.scaling_factors[i]

                self.lora_B[r_str].weight.data.copy_(b_init.to(self.lora_B[r_str].weight.dtype))

                # 索引递增，确保下一个分支拿到的是更小的奇异值对应的正交基
                current_idx += r_int

            # 6. 初始化可学习系数 alphas 为 1.0
            # 这样初始状态下，模型会保留 SVD 分解后的主要能量
            nn.init.ones_(self.alphas)

        # 显式清理大矩阵占用的内存
        del U, S, Vh, weight

    def reset_mr_parameters_standard(self):
        # Kaiming init for A, Zeros for B (ensures adapter starts as identity-zero)
        for a in self.lora_A.values():
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in self.lora_B.values():
            nn.init.zeros_(b.weight)
        # Initialize learnable coefficients to 1.0
        nn.init.ones_(self.alphas)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # 1. Base model forward
        result = self.get_base_layer()(x, *args, **kwargs)

        # 2. Optimized Mr. LoRA forward
        # Apply dropout once to the input to save compute and improve consistency
        x_dropped = self.lora_dropout(x)
        
        # 3. Vectorized Multi-Rank Path
        # List comprehension is still necessary for ModuleDict access,
        # but torch.stack moves the subsequent math to a vectorized kernel.
        rank_outputs = torch.stack([
            self.lora_B[r_str](self.lora_A[r_str](x_dropped)) 
            for r_str in self.ranks_str
        ]) # Shape: [num_ranks, batch, seq, out_features]

        # Alternative to stack + sum
        combined_scale = self.alphas * self.scaling_factors
        mr_adapter = torch.einsum('rbsh,r->bsh', rank_outputs, combined_scale)

        return result + mr_adapter

    def merge(self, safe_merge: bool = False, adapter_names = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """

        base_layer = self.get_base_layer()
        if safe_merge:
            # Note that safe_merge will be slower than the normal merge
            # because of the copy operation.
            orig_weights = base_layer.weight.data.clone()
            delta_weight = self.get_delta_weight()
            orig_weights += delta_weight
            if not torch.isfinite(orig_weights).all():
                raise ValueError(
                    f"NaNs detected in the merged weights."
                )

            base_layer.weight.data = orig_weights
        else:
            delta_weight = self.get_delta_weight()
            base_layer.weight.data += delta_weight

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        weight = self.get_base_layer().weight
        delta_weight = self.get_delta_weight()
        weight.data -= delta_weight

    def get_delta_weight(self):
        """计算所有 rank 组合后的 ΔW"""
        device = self.alphas.device
        dtype = self.lora_A[self.ranks_str[0]].weight.dtype
        out_features, in_features = self.get_base_layer().weight.shape

        # 初始化一个全零的 ΔW
        total_delta_w = torch.zeros(out_features, in_features, device=device, dtype=dtype)

        for i, r_str in enumerate(self.ranks_str):
            # W = B @ A
            delta_w = self.lora_B[r_str].weight @ self.lora_A[r_str].weight
            # Apply alpha and pre-computed scaling
            total_delta_w += self.alphas[i] * self.scaling_factors[i] * delta_w

        return total_delta_w
