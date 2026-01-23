from dataclasses import dataclass, field
from typing import Optional, List, Union
from peft.config import PeftConfig


@dataclass
class MrLoraConfig(PeftConfig):
    ranks: List[int] = field(default_factory=lambda: [32, 16, 8, 4, 2], metadata={"help": "List of ranks for multi-rank adaptation"})
    target_modules: Optional[Union[List[str], str]] = field(default=None, metadata={"help": "List of module names to apply Mr. LoRA to"})
    lora_alpha: int = field(default=16, metadata={"help": "The alpha parameter for LoRA scaling"})
    lora_dropout: float = field(default=0.0, metadata={"help": "The dropout probability for LoRA layers"})
    bias: str = field(default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(default=None, metadata={"help": "List of modules apart from LoRA layers to be set as trainable"})

    def __post_init__(self):
        self.peft_type = "MR_LORA"
