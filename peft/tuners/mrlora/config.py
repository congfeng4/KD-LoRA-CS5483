from dataclasses import dataclass, field
from typing import Optional, List
from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class MrLoraConfig(PeftConfig):
    """
    你的新 LoRA 方法配置
    """
    # 继承的基础配置
    auto_mapping: Optional[dict] = field(
        default=None,
        metadata={"help": "Auto mapping for model conversion"}
    )

    # 你的特定参数
    r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Target modules to apply LoRA"}
    )
    lora_dropout: float = field(default=0.0, metadata={"help": "Dropout rate"})
    bias: str = field(default="none", metadata={"help": "Bias type"})

    # 你的创新参数（如果有）
    my_new_param: float = field(default=0.1, metadata={"help": "你的新参数"})

    def __post_init__(self):
        # 必须指定 PEFT 类型
        self.peft_type = PeftType.MY_LORA  # 需要添加到 PeftType 枚举
        super().__post_init__()