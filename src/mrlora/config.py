from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple
from peft.config import PeftConfig


@dataclass
class MrLoraConfig(PeftConfig):
    init_weights: str = 'standard'
    total_rank: int = field(default=8,
                             metadata={"help": "Total rank for multi-rank adaptation"})
    target_modules: Optional[Union[List[str], str]] = field(default=None, metadata={
        "help": "List of module names to apply Mr. LoRA to"})
    lora_alpha: int = field(default=16, metadata={"help": "The alpha parameter for LoRA scaling"})
    lora_dropout: float = field(default=0.0, metadata={"help": "The dropout probability for LoRA layers"})
    bias: str = field(default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(default=None, metadata={
        "help": "List of modules apart from LoRA layers to be set as trainable"})
    use_rslora: bool = field(default=False, metadata={
        "help": "When True, uses rank-stabilized scaling (lora_alpha/sqrt(r) instead of lora_alpha/max(ranks))"})
    learn_coefficients: bool = field(default=False, metadata={
        "help": "When True, uses rank-stabilized scaling (lora_alpha/sqrt(r) instead of lora_alpha/max(ranks))"})

    layer_replication: Optional[List[Tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using LoRA to effectively expand a transformer model to a larger size by repeating some layers. "
                "The transformation handles models (currently Llama, Bert or Falcon compatible architectures) with "
                "a module list in the model which it modifies to expand the number of modules. "
                "Base weights are shared so the memory usage is close to the original model. The intended use is these base weights "
                "remain fixed during finetuning but each layer has a separate LoRA adapter so the layers can be specialed via "
                "the adapter layers fit during fine tuning."
                "The format is a list of [start, end) pairs which specify the layer ranges to stack. For example:\n"
                "   Original model has 5 layers labelled by their position in the model: `[0, 1, 2, 3, 4]`\n"
                "   layer_replication: `[[0, 4], [2, 5]]`\n"
                "   Final model will have this arrangement of original layers: `[0, 1, 2, 3, 2, 3, 4]`\n"
                "This format is based on what is used for pass-through merges in mergekit. It makes it simple to select sequential "
                "ranges of a model and stack them while reusing layers at either end of each sequence."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = "MR_LORA"
