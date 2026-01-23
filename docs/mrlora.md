To implement **Mr. LoRA (Multi-Rank LoRA)** within the Hugging Face `peft` framework, we must define a custom adapter that supports multiple low-rank decompositions () with learnable scaling coefficients ().

Below is the implementation partitioned into the requested file structure, followed by the modified training script.

---

### 1. `config.py`

This defines the configuration class inheriting from `PeftConfig`. We include `ranks` (a list like `[32, 16, 8, 4, 2]`) instead of a single integer `r`.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Union
from peft.utils import PeftType
from peft.config import PeftConfig

@dataclass
class MrLoRAConfig(PeftConfig):
    ranks: List[int] = field(default_factory=lambda: [32, 16, 8, 4, 2], metadata={"help": "List of ranks for multi-rank adaptation"})
    target_modules: Optional[Union[List[str], str]] = field(default=None, metadata={"help": "List of module names to apply Mr. LoRA to"})
    lora_alpha: int = field(default=16, metadata={"help": "The alpha parameter for LoRA scaling"})
    lora_dropout: float = field(default=0.0, metadata={"help": "The dropout probability for LoRA layers"})
    bias: str = field(default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(default=None, metadata={"help": "List of modules apart from LoRA layers to be set as trainable"})

    def __post_init__(self):
        self.peft_type = "MR_LORA"

```

---

### 2. `layer.py`

This implements the core logic: .

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer

class MrLoRALayer(nn.Module, LoraLayer):
    def __init__(self, in_features, out_features, ranks, lora_alpha, lora_dropout, **kwargs):
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer=kwargs.get("base_layer"))
        
        self.ranks = ranks
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(ranks) # Scaling relative to max rank
        
        # Multi-rank components
        self.lora_A = nn.ModuleList([nn.Linear(in_features, r, bias=False) for r in ranks])
        self.lora_B = nn.ModuleList([nn.Linear(r, out_features, bias=False) for r in ranks])
        # Learnable coefficients alpha_i
        self.alphas = nn.Parameter(torch.randn(len(ranks)))
        
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.reset_mr_parameters()

    def reset_mr_parameters(self):
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a.weight, a=5**0.5)
        for b in self.lora_B:
            nn.init.zeros_(b.weight)
        nn.init.normal_(self.alphas)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # Base model forward
        result = self.get_base_layer()(x, *args, **kwargs)
        
        # Mr. LoRA forward: sum(alpha_i * B_i(A_i(x)))
        x = x.to(self.lora_A[0].weight.dtype)
        
        mr_adapter = 0
        for i in range(len(self.ranks)):
            out = self.lora_B[i](self.lora_A[i](self.lora_dropout(x)))
            mr_adapter += self.alphas[i] * out
            
        return result + mr_adapter * self.scaling

class MrLoRALinear(MrLoRALayer):
    def __init__(self, base_layer, ranks, lora_alpha, lora_dropout, **kwargs):
        super().__init__(base_layer.in_features, base_layer.out_features, ranks, lora_alpha, lora_dropout, base_layer=base_layer)

```

---

### 3. `model.py`

This integrates the layers into the `PeftModel` structure.

```python
from peft.tuners.tuners_utils import BaseTuner
from .layer import MrLoRALinear, MrLoRALayer

class MrLoRAModel(BaseTuner):
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def _prepare_adapter_config(self, peft_config, model_config):
        return peft_config

    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent, **optional_kwargs):
        # Replace Linear with MrLoRALinear
        if isinstance(target, torch.nn.Linear):
            new_module = MrLoRALinear(
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

```

---

### 4. Modified `BERT_LoRA.py` Example

To use Mr. LoRA, we modify your provided script to import these custom classes.

```python
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
# Import Mr. LoRA components
# from mrlora.config import MrLoRAConfig
# from mrlora.model import MrLoRAModel

from utils import *

def main(args):
    # Determine labels
    num_labels = get_num_labels(args)

    # Load dataset and tokenizer
    dataset = load_dataset('glue', args.task, cache_dir=args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_datasets = dataset.map(tokenize_function(args, tokenizer), batched=True)

    # Prepare model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # 1. Define Mr. LoRA Config instead of standard LoraConfig
    # Setting ranks as 32, 16, 8, 4, 2 per your requirement
    mr_lora_config = MrLoRAConfig(
        ranks=[32, 16, 8, 4, 2],
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=args.lora_dropout,
        task_type="SEQ_CLS"
    )

    # 2. Apply Mr. LoRA
    # Using the custom tuner:
    model_lora = MrLoRAModel(model, mr_lora_config, "default")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=32,
        num_train_epochs=args.num_train_epochs,
        logging_strategy="steps", # Changed from "no" to see progress
        logging_steps=10
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics(args)
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./dataset")
    parser.add_argument("--model_name", type=str, default="./models/bert-base-uncased")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument('--task', type=str, default="wnli")
    main(parser.parse_args())

```

Would you like me to help you refine the initialization strategy for the learnable  coefficients to improve convergence?