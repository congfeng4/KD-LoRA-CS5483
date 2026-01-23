from .config import MrLoraConfig
from .layer import MrLoraLayer, Linear
from .model import MrLoraModel
from peft.utils import register_peft_method

__all__ = ["MrLoraConfig", "MrLoraLayer", "MrLoraModel", "Linear"]

register_peft_method(
    name="mrlora", config_cls=MrLoraConfig, model_cls=MrLoraModel, prefix="mrlora_", is_mixed_compatible=False
)
