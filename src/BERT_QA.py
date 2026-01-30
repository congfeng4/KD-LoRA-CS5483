import argparse
import json
import shutil
import sys
from pathlib import Path
from addict import Addict
import functools
import evaluate
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    set_seed,
    EvalPrediction
)
from peft import get_peft_model
from utils import *

# Import specialized components from official HF QA logic
# Note: These usually reside in the same directory as run_qa.py
try:
    from trainer_qa import QuestionAnsweringTrainer
    from utils_qa import postprocess_qa_predictions
except ImportError:
    print("Error: This script requires 'trainer_qa.py' and 'utils_qa.py' from the HF Transformers examples.")
    sys.exit(1)

# Suppress warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

RANK_VALUES = [8, 16, 32, 64]
seed_list = [42, 123, 2024]
QA_TASKS = ["squad-v1.1", "squad-v2"]
BENCH = "QA"


# --- HELPER FUNCTIONS FOR PREPROCESSING (Aligned with run_qa.py) ---

# Distillation Trainer
class QADistillTrainer(QuestionAnsweringTrainer):
    teacher_logits: torch.Tensor

    def compute_loss(self, model, inputs, return_outputs=False):
        idx = inputs.pop("idx").cpu()
        outputs = model(**inputs)
        s_start, s_end = outputs.start_logits, outputs.end_logits
        dtype = s_start.dtype
        device = s_start.device
        t_start = self.teacher_logits[0][idx].to(dtype).to(s_start.device)
        t_end = self.teacher_logits[1][idx].to(dtype).to(s_end.device)

        loss = (F.mse_loss(s_start, t_start) + F.mse_loss(s_end, t_end)) / 2.0
        return (loss, outputs) if return_outputs else loss


def prepare_train_features(examples, tokenizer, max_seq_length, doc_stride):
    # Standard HF QA preprocessing
    pad_on_right = tokenizer.padding_side == "right"
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, max_seq_length, doc_stride):
    pad_on_right = tokenizer.padding_side == "right"
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# --- PIPELINE CLASS ---

class QADistillPipeline:
    def __init__(self, **kwargs):
        self.args = Addict(kwargs)
        self.args.bench = BENCH
        self.dir = Path(self.args.dir_name)
        self.metric = evaluate.load("squad_v2" if self.args.task == "squad-v2" else "squad")

        self.training_params = dict(
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=self.args.train_batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            num_train_epochs=self.args.num_train_epochs,
            weight_decay=self.args.weight_decay,
            load_best_model_at_end=True,
            remove_unused_columns=False  # Important for distillation 'idx'
        )

    @functools.lru_cache(maxsize=None)
    def load_dataset_and_tokenize(self, model_name, is_student=False):
        # Load Raw
        if self.args.from_disk:
            raw_datasets = load_from_disk(os.path.join(self.args.dataset_path, self.args.task))
        else:
            raw_datasets = load_dataset(self.args.task)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Prepare Train
        train_dataset = raw_datasets["train"].map(
            lambda x: prepare_train_features(x, tokenizer, self.args.max_length, self.args.doc_stride),
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing train"
        )

        # Prepare Eval
        eval_examples = raw_datasets["validation"]
        eval_dataset = eval_examples.map(
            lambda x: prepare_validation_features(x, tokenizer, self.args.max_length, self.args.doc_stride),
            batched=True,
            remove_columns=eval_examples.column_names,
            desc="Tokenizing validation"
        )

        if is_student:
            # Add index for alignment during distillation
            train_dataset = train_dataset.map(lambda _, idx: {"idx": idx}, with_indices=True)

        return train_dataset, eval_dataset, eval_examples, tokenizer, raw_datasets

    def get_post_process_fn(self, eval_examples, tokenizer):
        def post_processing_function(examples, features, predictions, stage="eval"):
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                version_2_with_negative=(self.args.task == "squad_v2"),
                n_best_size=20,
                max_answer_length=30,
                null_score_diff_threshold=0.0,
                output_dir=self.args.dir_name,
                prefix=stage,
            )
            # Format for metric
            if self.args.task == "squad-v2":
                formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in
                                         predictions.items()]
            else:
                formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

        return post_processing_function

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def run_teacher_fft(self):
        args = self.args
        print('Begin Teacher FFT...')
        teacher_fft_dir = self.dir / 'fft' / self.args.task
        teacher_fft_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = teacher_fft_dir / 'metrics.json'
        ckpt_dir = str(teacher_fft_dir / "ckpt")
        if metrics_file.exists():
            return

        print('Preparing teacher FFT...')
        train_ds, eval_ds, eval_examples, tokenizer, raw_datasets = self.load_dataset_and_tokenize(
            self.args.teacher_model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(self.args.teacher_model_name)

        training_args = TrainingArguments(output_dir=ckpt_dir,
                                          learning_rate=self.args.teacher_learning_rate,
                                          **self.training_params)
        print('Loaded dataset & model', '#train', len(train_ds), '#eval', len(eval_ds),
              '#param', get_trainable_param_count(model))

        callback = MemoryTrackingCallback()
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            post_process_function=self.get_post_process_fn(eval_examples, tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[callback],
        )

        trainer.train()
        results = trainer.evaluate()

        # Save soft labels for distillation
        predict_examples = raw_datasets["test"]
        predictions = trainer.predict(train_ds, predict_examples)
        if trainer.is_world_process_zero():
            torch.save(predictions.predictions, teacher_fft_dir / 'teacher_soft_labels.pth')
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            shutil.rmtree(ckpt_dir)
            print(f"Teacher FFT Results: {results}")

        if hasattr(trainer, "accelerator"):
            trainer.accelerator.wait_for_everyone()

        model.to('cpu')
        del model
        clear_gpu_memory()
        print('Teacher FFT is done.', args.task, args.teacher_model_name)

    def run_student_lora(self):
        print("Begin Student Distill + LoRA...")
        student_dir = self.dir / 'kd-lora' / self.args.task
        student_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = student_dir / 'ckpt'
        metrics_file = student_dir / 'metrics.json'
        if metrics_file.exists():
            return
        # Load teacher labels
        teacher_fft_dir = self.dir / 'fft' / self.args.task
        teacher_logits = torch.load(teacher_fft_dir / 'teacher_soft_labels.pth')

        train_ds, eval_ds, eval_examples, tokenizer, _ = self.load_dataset_and_tokenize(self.args.student_model_name,
                                                                                        is_student=True)

        model = AutoModelForQuestionAnswering.from_pretrained(self.args.student_model_name)
        # Apply LoRA logic (assumes utility function exists as in your original snippet)
        peft_config = get_peft_config(self.args, self.args.student_model_name, self.args.peft)
        model = get_peft_model(model, peft_config)
        callback = MemoryTrackingCallback()

        training_args = TrainingArguments(output_dir=str(ckpt_dir),
                                          learning_rate=self.args.student_learning_rate,
                                          **self.training_params)

        trainer = QADistillTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            post_process_function=self.get_post_process_fn(eval_examples, tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[callback],
        )
        trainer.teacher_logits = teacher_logits

        trainer.train()
        results = trainer.evaluate()
        if trainer.is_world_process_zero():
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            shutil.rmtree(ckpt_dir)
            print(f"Student LoRA Results: {results}")

        if hasattr(trainer, "accelerator"):
            trainer.accelerator.wait_for_everyone()

        model.to('cpu')
        del model
        clear_gpu_memory()

    def run_teacher_lora(self):
        print("Begin Teacher LoRA...")
        student_dir = self.dir / 'lora' / self.args.task
        student_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = student_dir / 'ckpt'
        metrics_file = student_dir / 'metrics.json'
        if metrics_file.exists():
            return
        train_ds, eval_ds, eval_examples, tokenizer, _ = self.load_dataset_and_tokenize(self.args.student_model_name,
                                                                                        is_student=False)
        model = AutoModelForQuestionAnswering.from_pretrained(self.args.student_model_name)
        # Apply LoRA logic (assumes utility function exists as in your original snippet)
        peft_config = get_peft_config(self.args, self.args.student_model_name, self.args.peft)
        print('peft_config', peft_config)
        model = get_peft_model(model, peft_config)

        training_args = TrainingArguments(output_dir=str(ckpt_dir),
                                          learning_rate=self.args.student_learning_rate,
                                          **self.training_params)

        callback = MemoryTrackingCallback()
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            post_process_function=self.get_post_process_fn(eval_examples, tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[callback],
        )
        trainer.train()
        results = trainer.evaluate()
        if trainer.is_world_process_zero():
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            shutil.rmtree(ckpt_dir)
            print(f"Teacher LoRA Results: {results}")

        if hasattr(trainer, "accelerator"):
            trainer.accelerator.wait_for_everyone()

        model.to('cpu')
        del model
        clear_gpu_memory()


# --- MAIN EXECUTION ---

# Main loops similar to BERT_Distill_LoRA.py
def main_teacher_fft(args):
    for seed in seed_list:
        for task in QA_TASKS:
            for model_family in MODEL_FAMILY.keys():
                set_seed(seed)
                config = args.__dict__.copy()
                config['model_family'] = model_family
                config['task'] = task
                config['seed'] = seed
                add_model_name_to_config(model_family, config)
                pipe = QADistillPipeline(**config)
                try:
                    pipe.run_teacher_fft()
                except Exception as e:
                    print(e)
                    raise e
    print('All finish')


def main_lora(args, is_student: bool):
    for rank in RANK_VALUES:
        for seed in seed_list:
            for task in QA_TASKS:
                for model_family in MODEL_FAMILY.keys():
                    for peft_method in PEFT_FAMILY:
                        # Set alpha = 16 (fixed) as per our experimental setup
                        set_seed(seed)
                        config = args.__dict__.copy()
                        config['model_family'] = model_family
                        config['task'] = task
                        config['peft'] = peft_method
                        config['seed'] = seed
                        config['rank'] = rank
                        config['lora_alpha'] = 16

                        # For MrLoRA, generate ranks list from highest rank down to 1
                        # unless user provided custom lora_ranks (non-default)
                        if 'mrlora' in peft_method:
                            config['lora_ranks'] = generate_mrlora_ranks(rank)

                        add_model_name_to_config(model_family, config)
                        pipe = QADistillPipeline(**config)
                        try:
                            if is_student:
                                pipe.run_student_lora()
                            else:
                                pipe.run_teacher_lora()
                        except Exception as e:
                            print(e)
                            raise e
    print('All finish')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_name", type=str, default="./models/bert-base-uncased",
                        help="Name of the teacher model")
    parser.add_argument("--student_model_name", type=str, default="./models/distilbert-base-uncased",
                        help="Name of the student model")

    parser.add_argument("--task", type=str, default=QA_TASKS[0], choices=QA_TASKS)
    parser.add_argument("--dataset_path", type=str, default="./dataset")
    parser.add_argument("--from_disk", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dir_name", type=str, default="./results_qa")
    parser.add_argument('--type', '-t', type=int, choices=(0, 1, 2, 3),
                        help='0 => fft, 1 => student-lora, 2 => teacher-lora')
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout rate for LoRA layers")
    parser.add_argument('--use_rslora', action='store_true',
                        help='Use rank-stabilized scaling for MrLoRA (lora_alpha/sqrt(r) instead of lora_alpha/max(ranks))')
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA matrices")
    parser.add_argument('--peft', type=str, default="lora", choices=tuple(PEFT_FAMILY), help="PEFT method name")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument("--teacher_learning_rate", type=float, default=5e-5, help="Learning rate for the teacher model")
    parser.add_argument("--student_learning_rate", type=float, default=5e-4, help="Learning rate for the student model")

    args_cmd = parser.parse_args()

    if args_cmd.type == 0:
        main_teacher_fft(args_cmd)
    elif args_cmd.type == 1:
        main_lora(args_cmd, is_student=True)
    elif args_cmd.type == 2:
        main_lora(args_cmd, is_student=False)
    else:
        raise ValueError(f"Unknown command {args_cmd.type}")
