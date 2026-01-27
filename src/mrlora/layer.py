import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer


class MrLoraLayer(nn.Module, LoraLayer):
    def __init__(self, in_features, out_features, ranks, lora_alpha, lora_dropout, use_rslora=False, **kwargs):
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer=kwargs.get("base_layer"))

        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(ranks)  # Scaling relative to max rank (used when use_rslora=False)
        self.use_rslora = use_rslora
        self.ranks_int = ranks  # Store integer ranks for sqrt calculation

        # Multi-rank components
        self.lora_A = nn.ModuleDict({str(r): nn.Linear(in_features, r, bias=False) for r in ranks})
        self.lora_B = nn.ModuleDict({str(r): nn.Linear(r, in_features, bias=False) for r in ranks})
        # Learnable coefficients alpha_i
        self.alphas = nn.Parameter(torch.randn(len(ranks)))
        self.ranks = list(map(str, ranks))

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.reset_mr_parameters()

    def reset_mr_parameters(self):
        for a in self.lora_A.values():
            nn.init.kaiming_uniform_(a.weight, a=5 ** 0.5)
        for b in self.lora_B.values():
            nn.init.zeros_(b.weight)
        nn.init.normal_(self.alphas)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # Base model forward
        result = self.get_base_layer()(x, *args, **kwargs)

        # Mr. LoRA forward: sum(alpha_i * B_i(A_i(x)))
        # x = x.to(self.lora_A['0'].weight.dtype)

        mr_adapter = 0
        for i, r in enumerate(self.ranks):
            out = self.lora_B[r](self.lora_A[r](self.lora_dropout(x)))
            if self.use_rslora:
                # Rank-stabilized scaling: lora_alpha / sqrt(r)
                scaling = self.lora_alpha / math.sqrt(self.ranks_int[i])
                mr_adapter += self.alphas[i] * out * scaling
            else:
                mr_adapter += self.alphas[i] * out

        if not self.use_rslora:
            # Original scaling: lora_alpha / max(ranks)
            mr_adapter = mr_adapter * self.scaling
        
        return result + mr_adapter


class MrLoraLinear(MrLoraLayer):
    def __init__(self, base_layer, ranks, lora_alpha, lora_dropout, use_rslora=False, **kwargs):
        super().__init__(base_layer.in_features, base_layer.out_features, ranks, lora_alpha, lora_dropout,
                         use_rslora=use_rslora, base_layer=base_layer)
