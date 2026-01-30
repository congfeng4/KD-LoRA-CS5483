import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path
from addict import Addict
import functools
import evaluate
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    set_seed,
    EvalPrediction, AutoConfig, PreTrainedTokenizerFast, DataCollatorWithPadding, training_args
)
from peft import get_peft_model
from utils import *
from transformers import default_data_collator # Ensure this is imported
import logging

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
logger = logging.getLogger(__name__)


# --- HELPER FUNCTIONS FOR PREPROCESSING (Aligned with run_qa.py) ---

# Distillation Trainer
class QADistillTrainer(QuestionAnsweringTrainer):
    teacher_logits: torch.Tensor
    alpha: float = 0.5

    def compute_loss(self, model, inputs, return_outputs=False):
        idx = inputs.pop("idx").cpu()
        outputs = model(**inputs)
        s_start, s_end = outputs.start_logits, outputs.end_logits
        dtype = s_start.dtype
        t_start = self.teacher_logits[0][idx].to(dtype).to(s_start.device)
        t_end = self.teacher_logits[1][idx].to(dtype).to(s_end.device)
        distill_loss = (F.mse_loss(s_start, t_start) + F.mse_loss(s_end, t_end)) / 2.0
        qa_loss = outputs.loss
        loss = self.alpha * qa_loss + (1 - self.alpha) * distill_loss
        return (loss, outputs) if return_outputs else loss


def train_question_answering(args, raw_datasets, model, tokenizer, training_args, output_dir, *,
                             variant, teacher_soft_labels=None,
                             returns_predictions=False):
    # Set seed before initializing model.
    set_seed(args.seed)
    args.version_2_with_negative = 'v2' in args.task
    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )
        
    column_names = raw_datasets["train"].column_names # train == validation
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            if tokenizer.cls_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.cls_token_id)
            elif tokenizer.bos_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.bos_token_id)
            else:
                cls_index = 0

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    train_dataset = raw_datasets["train"]

    # Create train feature from dataset
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    eval_examples = raw_datasets["validation"]

    # Validation Feature Creation
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": str(k), "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": str(k), "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": str(ex["id"]), "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load('squad_v2' if 'v2' in args.task else 'squad', cache_dir=args.cache_dir)

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    callback = MemoryTrackingCallback()

    # Initialize our Trainer
    trainer_cls = QADistillTrainer if teacher_soft_labels is not None else QuestionAnsweringTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        processing_class=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        callbacks=[callback],
    )
    if teacher_soft_labels is not None:
        trainer.teacher_soft_labels = teacher_soft_labels
            
    train_result = trainer.train()
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        processing_class=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        callbacks=[callback],
    )
    metrics = trainer.evaluate()

    # Evaluation
    logger.info("*** Evaluate ***")
    patch_results(metrics, args, train_result, variant)

    if returns_predictions:
        predictions = trainer.predict(train_dataset, raw_datasets['train']).predictions
        return metrics, trainer, predictions

    return metrics, trainer


# --- PIPELINE CLASS ---

class QADistillPipeline:

    def __init__(self, **kwargs):
        self.args = Addict(kwargs)
        self.args.bench = BENCH
        self.dir = Path(self.args.dir_name)
        self.metric = evaluate.load("squad_v2" if self.args.task == "squad-v2" else "squad")

        self.training_params = dict(
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=self.args.train_batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            num_train_epochs=self.args.num_train_epochs,
            weight_decay=self.args.weight_decay,
            load_best_model_at_end=True,
            # remove_unused_columns=False  # Important for distillation 'idx'
        )

    @functools.lru_cache(maxsize=None)
    def load_dataset(self):
        dataset_path = os.path.join(self.args.dataset_path, self.args.task)
        return load_from_disk(dataset_path)

    def load_model(self, model_name_or_path):
        config = AutoConfig.from_pretrained(
            model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            config=config,
        )
        return model, tokenizer

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
        raw_datasets = self.load_dataset()
        model, tokenizer = self.load_model(args.teacher_model_name)
        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
            learning_rate=args.student_learning_rate,
            **self.training_params,
        )
        results, trainer, predictions = train_question_answering(self.args, raw_datasets, model, tokenizer,
                                                                 training_args, ckpt_dir,
                                                                 variant='fft',
                                                                 returns_predictions=True)

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
        args = self.args
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

        raw_datasets = self.load_dataset()
        model, tokenizer = self.load_model(args.student_model_name)

        # Apply LoRA logic (assumes utility function exists as in your original snippet)
        peft_config = get_peft_config(self.args, self.args.student_model_name, self.args.peft)
        model = get_peft_model(model, peft_config)
        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
            learning_rate=args.student_learning_rate,
            **self.training_params,
        )
        results, trainer = train_question_answering(self.args, raw_datasets, model, tokenizer, training_args, ckpt_dir,
                                                    variant='kd-lora',
                                                    teacher_soft_labels=teacher_logits)

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
        args = self.args
        print("Begin Teacher LoRA...")
        student_dir = self.dir / 'lora' / self.args.task
        student_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = student_dir / 'ckpt'
        metrics_file = student_dir / 'metrics.json'
        if metrics_file.exists():
            return
        raw_datasets = self.load_dataset()
        model, tokenizer = self.load_model(args.student_model_name)
        # Apply LoRA logic (assumes utility function exists as in your original snippet)
        peft_config = get_peft_config(self.args, self.args.student_model_name, self.args.peft)
        print('peft_config', peft_config)
        model = get_peft_model(model, peft_config)

        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
            learning_rate=args.student_learning_rate,
            **self.training_params,
        )
        results, trainer = train_question_answering(self.args, raw_datasets, model, tokenizer, training_args, ckpt_dir,
                                                    variant='lora')

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
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dir_name", type=str, default="./results_qa")
    parser.add_argument('--type', '-t', type=int, choices=(0, 1, 2, 3), default=0,
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
    parser.add_argument('--cache_dir', type=str, default='.')
    parser.add_argument('--pad_to_max_length', type=int, default=1)
    parser.add_argument('--n_best_size', type=int, default=20)
    parser.add_argument('--max_answer_length', type=int, default=30)
    parser.add_argument('--preprocessing_num_workers', type=int, default=2)
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0)

    args_cmd = parser.parse_args()

    if args_cmd.type == 0:
        main_teacher_fft(args_cmd)
    elif args_cmd.type == 1:
        main_lora(args_cmd, is_student=True)
    elif args_cmd.type == 2:
        main_lora(args_cmd, is_student=False)
    else:
        raise ValueError(f"Unknown command {args_cmd.type}")
