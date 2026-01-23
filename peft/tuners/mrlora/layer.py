import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from peft.tuners.tuners_utils import BaseTunerLayer


class MrLoraLayer(BaseTunerLayer):
    """
    你的 LoRA 层实现
    """

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        self.r = kwargs.get("r", 8)
        self.lora_alpha = kwargs.get("lora_alpha", 16)
        self.scaling = self.lora_alpha / self.r

        # 初始化你的低秩矩阵
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # 标准 LoRA: W' = W + alpha/r * B*A
        # 你的创新可以在这里修改
        self.lora_A = nn.Parameter(torch.zeros(in_features, self.r))
        self.lora_B = nn.Parameter(torch.zeros(self.r, out_features))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 如果有 dropout
        self.lora_dropout = nn.Dropout(kwargs.get("lora_dropout", 0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础输出
        base_output = self.base_layer(x)

        # LoRA 分支
        x = self.lora_dropout(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling

        return base_output + lora_output

    def merge(self) -> None:
        """合并权重到基础层（用于推理优化）"""
        if self.merged:
            return

        # 计算合并后的权重
        delta_weight = (self.lora_B @ self.lora_A.T) * self.scaling
        self.base_layer.weight.data += delta_weight.T
        self.merged = True

    def unmerge(self) -> None:
        """解除合并"""
        if not self.merged:
            return
        # 恢复原始权重...
        self.merged = False


class Linear(MyLoraLayer, nn.Linear):
    """具体的 Linear 层实现"""

    def __init__(self, base_layer: nn.Linear, **kwargs):
        # 初始化父类
        MyLoraLayer.__init__(self, base_layer, **kwargs)
        # 保持 Linear 的接口
        nn.Linear.__init__(self, base_layer.in_features, base_layer.out_features)