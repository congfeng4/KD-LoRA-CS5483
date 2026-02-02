from .config import MrLoraConfig
from .layer import MrLoraLayer
from .model import MrLoraModel
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING

__all__ = ["MrLoraConfig", "MrLoraLayer", "MrLoraModel"]

PEFT_TYPE_TO_MODEL_MAPPING['MR_LORA'] = MrLoraModel
PEFT_TYPE_TO_CONFIG_MAPPING['MR_LORA'] = MrLoraConfig
PEFT_TYPE_TO_TUNER_MAPPING['MR_LORA'] = MrLoraModel
