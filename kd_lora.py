# %%
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F

glue_tasks = [
    "cola", "sst2", "mrpc", "qqp", "stsb",
    "mnli", "qnli", "rte", "wnli", "ax"
]

parser = argparse.ArgumentParser(description="Knowledge Distillation with LoRA-enhanced Student Model")

# Model arguments
parser.add_argument("--teacher_model_name", type=str, default="./models/bert-base-uncased",
                    help="Name of the teacher model")
parser.add_argument("--student_model_name", type=str, default="./models/distilbert-base-uncased",
                    help="Name of the student model")

# Dataset and training parameters
parser.add_argument("--dataset_path", type=str, default='./dataset', help="Path to the dataset")
parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for the classification task")
parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")

# LoRA parameters
parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA matrices")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA layers")
parser.add_argument('--task', type=str, default="wnli", choices=tuple(glue_tasks))

# Learning rates for teacher and student
parser.add_argument("--teacher_learning_rate", type=float, default=5e-5, help="Learning rate for the teacher model")
parser.add_argument("--student_learning_rate", type=float, default=5e-5, help="Learning rate for the student model")

args = parser.parse_args()

# Now you can access them as usual:
print(args.teacher_model_name)

# %%
# Step 1: Fine-tune a Teacher Model
print(f"Fine-tuning the teacher model: {args.teacher_model_name}")
teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_name, num_labels=args.num_labels)
teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)

# %%
teacher_training_args = TrainingArguments(
    output_dir="./teacher_results/" + args.task,
    learning_rate=args.teacher_learning_rate,
    per_device_train_batch_size=args.train_batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.01,
    eval_strategy="epoch",
)

# %%

# %%
from datasets import load_dataset

# 下载并加载 GLUE 的 WNLI 子集
teacher_dataset = load_dataset("glue", args.task, cache_dir=args.dataset_path)

# %%
teacher_dataset

# %%
# teacher_dataset = load_from_disk(args.dataset_path)
tokenized_teacher_dataset = teacher_dataset.map(
    lambda x: teacher_tokenizer(x["sentence1"], x["sentence2"], padding="max_length", truncation=True),
    batched=True
)

# %%
# Define trainer for teacher model
teacher_trainer = Trainer(
    model=teacher_model,
    args=teacher_training_args,
    train_dataset=tokenized_teacher_dataset["train"],
    eval_dataset=tokenized_teacher_dataset["validation"]
)
teacher_trainer.train()

# %%
# Save teacher model predictions (logits) as soft labels
teacher_logits = teacher_trainer.predict(tokenized_teacher_dataset["train"]).predictions
teacher_soft_labels = torch.tensor(teacher_logits)

# %%
teacher_model.save_pretrained('./pretrained/bert-base-uncased-FFT-wnli')

# %%
teacher_soft_labels.shape

# %%
# Step 2: Initialize a Smaller Student Model with LoRA
print(f"Initializing student model: {args.student_model_name} with LoRA")
student_model = AutoModelForSequenceClassification.from_pretrained(args.student_model_name, num_labels=args.num_labels)
student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name)

# %%
lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.lora_alpha,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="SEQ_CLS"
)

# %%
# Run this to see the exact names of your layers
for name, module in student_model.named_modules():
    print(name)

# %%
# Apply LoRA configuration to the student model
student_model = get_peft_model(student_model, lora_config)

# %%
student_model

# %%
# Freeze all layers except LoRA parameters
for param in student_model.parameters():
    param.requires_grad = False
for name, param in student_model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True  # Only LoRA weights are trainable

# %%
# Step 3: Distillation from Teacher to Student
print("Starting knowledge distillation from teacher to student")
student_training_args = TrainingArguments(
    output_dir="./student_results/" + args.task,
    learning_rate=args.student_learning_rate,
    per_device_train_batch_size=args.train_batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.01,
    remove_unused_columns=False,
    eval_strategy="epoch",
)


# %%
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    # Compute the distillation loss with temperature scaling
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


# Define a custom training loop for distillation
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwds):
        labels = inputs.pop("labels")
        idx = inputs.pop('idx').long().cpu()
        outputs = model(**inputs)
        student_logits = outputs.logits
        teacher_logits = teacher_soft_labels[idx]  # Align teacher logits with batch size
        # teacher_logits = teacher_soft_labels[inputs["input_ids"].shape[0]]  # Align teacher logits with batch size
        teacher_logits = teacher_logits.to(student_logits.device)
        loss = distillation_loss(student_logits, teacher_logits, labels)
        return (loss, outputs) if return_outputs else loss


# %%
# Tokenize student dataset
tokenized_student_dataset = teacher_dataset.map(
    lambda x, idx: {**student_tokenizer(x["sentence1"], x["sentence2"], padding="max_length", truncation=True),
                    'idx': idx},
    batched=True, with_indices=True
)

# %%
tokenized_student_dataset['train'][0]['idx']

# %%
# Initialize Distillation Trainer
student_trainer = DistillationTrainer(
    model=student_model,
    args=student_training_args,
    train_dataset=tokenized_student_dataset["train"],
    eval_dataset=tokenized_student_dataset["validation"]
)

# Train student model with knowledge distillation
student_trainer.train()

# %%
# Evaluate student model
student_trainer.evaluate()

# %%
# Save the fine-tuned LoRA student model
output_dir = "./fine_tuned_student_model"
student_model.save_pretrained(output_dir)
student_tokenizer.save_pretrained(output_dir)
print(f"Student model saved to {output_dir}")

# %%
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
test_data = teacher_dataset['validation']  # 如果要生成提交文件，请换成 dataset["test"]
model = student_model


# 4. 预处理函数
def preprocess_function(examples):
    return student_tokenizer(examples["sentence1"], examples["sentence2"],
                             truncation=True, padding="max_length", max_length=128)


tokenized_test = test_data.map(preprocess_function, batched=True)
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 5. 推理循环
predictions = []
references = []

dataloader = DataLoader(tokenized_test, batch_size=16)

print("正在进行推理...")
for batch in tqdm(dataloader):
    inputs = {k: v.to(model.device) for k, v in batch.items() if k != "label"}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)
    predictions.extend(preds.cpu().numpy())
    references.extend(batch["label"].cpu().numpy())

# 6. 计算准确率 (仅适用于 validation)
correct = sum(1 for p, r in zip(predictions, references) if p == r)
accuracy = correct / len(references)
print(f"\nValidation Accuracy: {accuracy:.4f}")

# %% [markdown]
# Exactly the same as reported in the paper.
