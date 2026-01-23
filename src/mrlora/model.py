import torch
from peft.tuners.tuners_utils import BaseTuner
from .layer import MrLoraLinear, MrLoraLayer


class MrLoraModel(BaseTuner):

    def set_adapter(self, adapter_names):
        """设置当前激活的适配器。对于 MrLoRA，我们目前假设只有一个。"""
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for module in self.model.modules():
            if isinstance(module, MrLoraLinear):
                # 这里可以扩展多适配器逻辑，目前简单处理
                module.active_adapter = adapter_names[0]

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

    def _check_target_module_exists(self, peft_config, key):
        # Basic check if the current module name matches target_modules
        return any(target in key for target in peft_config.target_modules)

    def _mark_only_adapters_as_trainable(self, model: torch.nn.Module) -> None:
        """
        Marks only MrLoRA parameters (lora_A, lora_B, alphas) 
        and specifically requested modules (like classifier) as trainable.
        """
        # 1. Freeze everything first
        for p in model.parameters():
            p.requires_grad = False

        # 2. Unfreeze MrLoRA specific weights and the classification head
        config = self.peft_config[self.active_adapter]
        for n, p in model.named_parameters():
            # Unfreeze if it's part of our Multi-Rank adapters
            if "lora_" in n or "alphas" in n:
                p.requires_grad = True

            # Unfreeze modules_to_save (e.g., ['classifier'])
            if config.modules_to_save and any(m in n for m in config.modules_to_save):
                p.requires_grad = True

    def disable_adapter_layers(self):
        # Logic to skip the MrLoRA forward pass (use only base layer)
        for module in self.model.modules():
            if isinstance(module, MrLoraLinear):
                module.disable_adapters = True

    def enable_adapter_layers(self):
        # Logic to "turn on" the MrLoRA forward pass
        for module in self.model.modules():
            if isinstance(module, MrLoraLinear):
                module.disable_adapters = False
