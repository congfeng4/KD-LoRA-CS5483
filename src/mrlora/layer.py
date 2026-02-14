import math

import torch
import torch.nn as nn

from .config import MrLoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer


def generate_mrlora_ranks(total_rank: int):
    # TODO: Merge rank 1.
    # total_rank=8: 1,2,5
    assert total_rank % 2 == 0, total_rank

    highest_rank = total_rank//2 + 1
    ranks = []
    r = highest_rank
    while r > 1:
        ranks.append(r)
        r = r // 2

    ranks[-1] += 1
    assert sum(ranks) == total_rank, (ranks, total_rank)
    return ranks


class MrLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("mrlora_A", "mrlora_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("ranks_int", "ranks_str", "mrlora_lambdas", "mrlora_scaling_factors")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.kwargs = kwargs
        self.ranks_int = []
        self.ranks_str = []
        self.mrlora_A = nn.ModuleDict()
        self.mrlora_B = nn.ModuleDict()
        self.mrlora_lambdas = nn.ParameterDict()
        self.mrlora_scaling_factors = nn.ParameterDict()

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, mrlora_config):
        if mrlora_config.total_rank <= 0:
            raise ValueError(f"`total_rank` should be a positive integer value but the value passed is {mrlora_config.total_rank}")
        if mrlora_config.total_rank % 2 != 0:
            raise ValueError(f"`total_rank` should be an even integer value but the value passed is {mrlora_config.total_rank}")

        self.ranks_int = generate_mrlora_ranks(mrlora_config.total_rank)
        # print('self.rank_int', self.ranks_int)
        # print('total_rank', mrlora_config.total_rank)
        self.ranks_str = list(map(str, self.ranks_int))
        mrlora_A = nn.ModuleDict()
        mrlora_B = nn.ModuleDict()
        for r_str, r_int in zip(self.ranks_str, self.ranks_int):
            mrlora_A[r_str] = nn.Linear(in_features=self.in_features, out_features=r_int, bias=mrlora_config.use_bias)
            mrlora_B[r_str] = nn.Linear(in_features=r_int, out_features=self.out_features, bias=mrlora_config.use_bias)

        # print('mrlora_config.use_lcoef', mrlora_config.use_lcoef)
        mrlora_config.use_lcoef = False
        self.mrlora_A.update(nn.ModuleDict(dict(default=mrlora_A)))
        self.mrlora_B.update(nn.ModuleDict(dict(default=mrlora_B)))
        self.mrlora_lambdas.update(dict(default=nn.Parameter(torch.ones(len(self.ranks_int)),
                                           requires_grad=mrlora_config.use_lcoef)))
        # Since we have multiple ranks in one layer, if the n
        # if mrlora_config.use_rslora:
            # RS-LoRA: alpha / sqrt(r)
        # scalings = [math.sqrt(r) for r in self.ranks_int]
            # print('use_rslora', scalings)
        scalings = [ r for r in self.ranks_int]

        # scalings = [1 for r in self.ranks_int]
        print('scalings', scalings)

        # print('scalings', scalings)
        self.mrlora_scaling_factors.update(dict(
            default=torch.nn.Parameter(torch.tensor(scalings), requires_grad=False)))
        
        lora_dropout = mrlora_config.lora_dropout
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout = lora_dropout_layer

        self.reset_mr_parameters(adapter_name, use_olora=mrlora_config.use_olora)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_mr_parameters(self, adapter_name, use_olora: bool):
        nn.init.ones_(self.mrlora_lambdas['default'])

        if use_olora:
            self.reset_mr_parameters_olora()
        else:
            self.reset_mr_parameters_lora()

    @torch.no_grad()
    def reset_mr_parameters_olora(self):
        """
        实现基于 OLoRA 思想的 SVD 能量切片初始化：
        1. 对原始权重 W 进行 SVD 分解。
        2. 按奇异值从高到低，将对应的正交基分配给 Rank 4, 2, 1 等矩阵。
        """
        mrlora_B, mrlora_A = self.mrlora_B['default'], self.mrlora_A['default']
        combined_scale = self.mrlora_lambdas['default'] * self.mrlora_scaling_factors['default']

        # 1. 获取基础层的权重数据
        # 获取 base_layer 权重 [out_features, in_features]
        weight = self.get_base_layer().weight.data.float()

        # 2. 执行 SVD 分解
        # U: [out_features, K], S: [K], Vh: [K, in_features]
        # 这里 K = min(out_features, in_features)
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        # 3. 按照你的计划进行“能量切片”分配
        current_idx = 0
        # TODO: W -= QR!!
        for i, r_str in enumerate(self.ranks_str):
            r_int = self.ranks_int[i]

            # 在 SVD 结果中提取对应的分量
            # 取出第 current_idx 到 current_idx + r_int 个奇异向量
            u_slice = U[:, current_idx: current_idx + r_int]
            s_slice = S[current_idx: current_idx + r_int]
            vh_slice = Vh[current_idx: current_idx + r_int, :]

            # 4. 初始化 A：分配右奇异向量 (正交基)
            # mrlora_A 的形状通常是 [r, in_features]
            mrlora_A[r_str].weight.data.copy_(vh_slice.to(mrlora_A[r_str].weight.dtype))

            # 5. 初始化 B：分配左奇异向量并结合奇异值
            # 理论上 AB = u_slice @ diag(s_slice) @ vh_slice
            b_init = u_slice @ torch.diag(s_slice)

            mrlora_B[r_str].weight.data.copy_(b_init.to(mrlora_B[r_str].weight.dtype))

            # 索引递增，确保下一个分支拿到的是更小的奇异值对应的正交基
            current_idx += r_int
            # TODO: Consider the lambda & scalling!!
            self.get_base_layer().weight -= mrlora_B[r_str].weight @ mrlora_A[r_str].weight * combined_scale[i]

        # 显式清理大矩阵占用的内存
        del U, S, Vh, weight

    def reset_mr_parameters_lora(self):
        # print('reset_mr_parameters_lora')
        for a in self.mrlora_A['default'].values():
            nn.init.zeros_(a.weight)
        for b in self.mrlora_B['default'].values():
            nn.init.normal_(b.weight)


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
            mrlora_B, mrlora_A = self.mrlora_B['default'], self.mrlora_A['default']
            ranks_str = self.ranks_str
            # 2. Optimized Mr. LoRA forward
            # Apply dropout once to the input to save compute and improve consistency
            x_dropped = self.lora_dropout(x)

            # 3. Vectorized Multi-Rank Path
            # List comprehension is still necessary for ModuleDict access,
            # but torch.stack moves the subsequent math to a vectorized kernel.
            rank_outputs = torch.stack([
                mrlora_B[r_str](mrlora_A[r_str](x_dropped))
                for r_str in ranks_str
            ])  # Shape: [num_ranks, batch, seq, out_features]

            # Alternative to stack + sum
            combined_scale = self.mrlora_lambdas['default'] * self.mrlora_scaling_factors['default']
            delta_x = torch.einsum('rbsh,r->bsh', rank_outputs, combined_scale)
            # print('delta_x', delta_x.min(), delta_x.max())
            result += delta_x

        result = result.to(previous_dtype)
        # print('result', result)
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
        # print('Merge!!!')
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
        mrlora_B, mrlora_A = self.mrlora_B['default'], self.mrlora_A['default']
        combined_scales = self.mrlora_lambdas['default'] * self.mrlora_scaling_factors['default']

        # 初始化一个全零的 ΔW
        total_delta_w = torch.zeros_like(self.base_layer.weight)

        for i, r_str in enumerate(self.ranks_str):
            # W = B @ A
            delta_w = mrlora_B[r_str].weight @ mrlora_A[r_str].weight * combined_scales[i]
            total_delta_w += delta_w

        return total_delta_w

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "mrlora." + rep
