import torch
from torch import nn
from typing import Optional, Dict, Any
from peft.tuners.tuners_utils import BaseTuner
from .layer import Linear, MrLoraLayer
from .config import MrLoraConfig


class MrLoraModel(BaseTuner):
    """
    你的 LoRA 模型封装
    """
    prefix = "mrlora_"

    def __init__(self, model, config: MrLoraConfig, adapter_name: str = "default"):
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        """准备适配器配置"""
        if peft_config.target_modules is None:
            # 设置默认目标模块（通常是 attention 层）
            peft_config.target_modules = ["q_proj", "v_proj"]
        return peft_config

    def _create_new_module(self, lora_config: MrLoraConfig, adapter_name: str, target, **kwargs):
        """创建新的 LoRA 模块"""
        # 根据目标层类型创建对应的 LoRA 层
        if isinstance(target, nn.Linear):
            new_module = Linear(target, **kwargs)
        else:
            raise ValueError(f"Unsupported layer type: {type(target)}")

        return new_module

    def merge_adapter(self, safe_merge: bool = False, adapter_names: Optional[list] = None) -> None:
        """合并适配器权重"""
        # 实现合并逻辑
        for name, module in self.model.named_modules():
            if isinstance(module, MrLoraLayer):
                module.merge()

    def unmerge_adapter(self, adapter_names: Optional[list] = None) -> None:
        """解除合并"""
        for name, module in self.model.named_modules():
            if isinstance(module, MrLoraLayer):
                module.unmerge()

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        """检查目标模块是否存在"""
        return any(key.endswith(f".{target}") for target in lora_config.target_modules)