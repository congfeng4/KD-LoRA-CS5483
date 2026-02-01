import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer


def generate_mrlora_ranks(highest_rank):
    """Generate MrLoRA ranks list from highest_rank down to 1 by halving."""
    ranks = []
    r = highest_rank
    while r >= 1:
        ranks.append(r)
        r = r // 2

    return ranks


class MrLoraLayer(nn.Module, LoraLayer):
    def __init__(self, in_features, out_features, total_rank, lora_alpha, lora_dropout, use_rslora=True, **kwargs):
        nn.Module.__init__(self)
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

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        # 将 ΔW 加到 base_layer 的权重上
        delta_w = self.get_delta_weight()
        self.get_base_layer().weight.data += delta_w

    def unmerge(self):
        # 减去 ΔW 还原权重
        delta_w = self.get_delta_weight()
        self.get_base_layer().weight.data -= delta_w

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


class MrLoraLinear(MrLoraLayer):
    def __init__(self, base_layer, total_rank, lora_alpha, lora_dropout, use_rslora=False, **kwargs):
        super().__init__(base_layer.in_features, base_layer.out_features, total_rank, lora_alpha, lora_dropout,
                         use_rslora=use_rslora, base_layer=base_layer)
