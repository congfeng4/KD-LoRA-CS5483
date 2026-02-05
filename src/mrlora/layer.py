import math

import torch
import torch.nn as nn

from .config import MrLoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer


def generate_mrlora_ranks(highest_rank):
    """Generate MrLoRA ranks list from highest_rank down to 1 by halving."""
    ranks = []
    r = highest_rank
    while r >= 1:
        ranks.append(r)
        r = r // 2

    return ranks


class MrLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("mrlora_A", "mrlora_B", "mrlora_lambdas", "scaling_factors")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("ranks_int", "ranks_str", )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.kwargs = kwargs
        self.ranks_int = []
        self.ranks_str = []
        self.mrlora_A = nn.ParameterDict()
        self.mrlora_B = nn.ParameterDict()
        self.mrlora_lambdas = nn.ParameterDict()
        self.scaling_factors = nn.ParameterDict()
        self.learn_coefficients = False

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, mrlora_config):
        self.learn_coefficients = mrlora_config.learn_coefficients
        if mrlora_config.total_rank <= 0:
            raise ValueError(f"`total_rank` should be a positive integer value but the value passed is {mrlora_config.total_rank}")
        if mrlora_config.total_rank % 2 != 0:
            raise ValueError(f"`total_rank` should be an even integer value but the value passed is {mrlora_config.total_rank}")

        self.ranks_int = generate_mrlora_ranks(mrlora_config.total_rank)
        self.ranks_str = list(map(str, self.ranks_int))
        for r_str, r_int in zip(self.ranks_str, self.ranks_int):
            self.mrlora_A[r_str] = nn.Linear(in_features=self.in_features, out_features=r_int, bias=False)
            self.mrlora_B[r_str] = nn.Linear(in_features=r_int, out_features=self.out_features, bias=False)
        
        self.mrlora_lambdas.update(dict(lambdas=nn.Parameter(torch.ones(len(self.ranks_int)),
                                           requires_grad=mrlora_config.learn_coefficients)))
        
        lora_dropout = mrlora_config.lora_dropout
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout = lora_dropout_layer

        # # 1. Pre-compute scaling factors to avoid math in the forward pass
        use_rslora = mrlora_config.use_rslora
        lora_alpha = mrlora_config.lora_alpha
        if use_rslora:
            # RS-LoRA: alpha / sqrt(r)
            scalings = [lora_alpha / math.sqrt(r) for r in self.ranks_int]
        else:
            # Standard: alpha / r (or alpha / max_r as per your original logic)
            # Standard LoRA usually uses alpha / r.
            scalings = [lora_alpha / r for r in self.ranks_int]

        self.scaling_factors.update(dict(
            factors=torch.nn.Parameter(torch.tensor(scalings), requires_grad=False)))

        self.reset_mr_parameters(adapter_name, init_weights=mrlora_config.init_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_mr_parameters(self, adapter_name, init_weights):
        if init_weights == 'standard':
            self.reset_mr_parameters_standard()
        elif init_weights == 'olora':
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
                # mrlora_A 的形状通常是 [r, in_features]
                self.mrlora_A[r_str].weight.data.copy_(vh_slice.to(self.mrlora_A[r_str].weight.dtype))

                # 5. 初始化 B：分配左奇异向量并结合奇异值
                # 理论上 AB = u_slice @ diag(s_slice) @ vh_slice
                # 为了抵消 forward 中的 scaling_factor，这里需要除以它
                b_init = u_slice @ torch.diag(s_slice)
                b_init = b_init / self.scaling_factors[i]

                self.mrlora_B[r_str].weight.data.copy_(b_init.to(self.mrlora_B[r_str].weight.dtype))

                # 索引递增，确保下一个分支拿到的是更小的奇异值对应的正交基
                current_idx += r_int

            # 6. 初始化可学习系数 alphas 为 1.0
            # 这样初始状态下，模型会保留 SVD 分解后的主要能量
            nn.init.ones_(self.mrlora_lambdas)

        # 显式清理大矩阵占用的内存
        del U, S, Vh, weight

    def reset_mr_parameters_standard(self):
        # Kaiming init for A, Zeros for B (ensures adapter starts as identity-zero)
        for a in self.mrlora_A.values():
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in self.mrlora_B.values():
            nn.init.zeros_(b.weight)
        # Initialize learnable coefficients to 1.0
        if self.learn_coefficients:
            nn.init.ones_(self.mrlora_lambdas)


class Linear(nn.Module, MrLoraLayer):

    def __init__(self,
                base_layer,
                adapter_name: str,
                mrlora_config: MrLoraConfig,
                **kwargs,
                ):
        super().__init__()
        MrLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, mrlora_config)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        previous_dtype = x.dtype
        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

            ranks_str = self.ranks_str
            # 2. Optimized Mr. LoRA forward
            # Apply dropout once to the input to save compute and improve consistency
            x_dropped = self.lora_dropout(x)

            # 3. Vectorized Multi-Rank Path
            # List comprehension is still necessary for ModuleDict access,
            # but torch.stack moves the subsequent math to a vectorized kernel.
            rank_outputs = torch.stack([
                self.mrlora_B[r_str](self.mrlora_A[r_str](x_dropped))
                for r_str in ranks_str
            ])  # Shape: [num_ranks, batch, seq, out_features]

            # Alternative to stack + sum
            combined_scale = self.mrlora_lambdas['lambdas'] * self.scaling_factors['factors']
            result += torch.einsum('rbsh,r->bsh', rank_outputs, combined_scale)

        result = result.to(previous_dtype)
        return result

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

    def get_delta_weight(self, adapter=None) -> torch.Tensor:
        """计算所有 rank 组合后的 ΔW"""
        device = self.mrlora_lambdas.device
        dtype = self.mrlora_A[self.ranks_str[0]].weight.dtype
        out_features, in_features = self.get_base_layer().weight.shape

        # 初始化一个全零的 ΔW
        total_delta_w = torch.zeros(out_features, in_features, device=device, dtype=dtype)

        for i, r_str in enumerate(self.ranks_str):
            # W = B @ A
            delta_w = self.mrlora_B[r_str].weight @ self.mrlora_A[r_str].weight
            # Apply alpha and pre-computed scaling
            total_delta_w += self.mrlora_lambdas[i] * self.scaling_factors[i] * delta_w

        return total_delta_w

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "mrlora." + rep
