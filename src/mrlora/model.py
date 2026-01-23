import torch
from peft.tuners.tuners_utils import BaseTuner
from .layer import MrLoraLinear, MrLoraLayer


class MrLoraModel(BaseTuner):
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def _prepare_adapter_config(self, peft_config, model_config):
        return peft_config

    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent, **optional_kwargs):
        # Replace Linear with MrLoRALinear
        if isinstance(target, torch.nn.Linear):
            new_module = MrLoraLinear(
                target, 
                ranks=peft_config.ranks, 
                lora_alpha=peft_config.lora_alpha, 
                lora_dropout=peft_config.lora_dropout
            )
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, old_module):
        setattr(parent, child_name, new_module)
        # Ensure base layer is accessible
        if hasattr(old_module, "base_layer"):
            new_module.base_layer = old_module.base_layer
        else:
            new_module.base_layer = old_module

    def __getattr__(self, name: str):
        return getattr(self.model, name)