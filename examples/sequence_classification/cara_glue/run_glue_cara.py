"""Single-run CaRA + GLUE fine-tuning for DeBERTaV3-base.

Mirrors the PSOFT protocol (arXiv 2505.11235):
    https://github.com/fei407/PSOFT/blob/master/1-NLU/script/deberta_v3_base_psoft-cola.sh
Uses HuggingFace Trainer with load_best_model_at_end for deterministic, paper-aligned results.

Example:
    python run_glue_cara.py --task mrpc --seed 42 \
        --output_dir results/cara_r8/mrpc/seed_42
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import CaraConfig, get_peft_model


TASK_CONFIG: dict[str, dict] = {
    "cola": {
        "text_cols": ("sentence",),
        "num_labels": 2,
        "is_regression": False,
        "metric_for_best_model": "matthews_correlation",
        "primary_metric": "matthews_correlation",
        "defaults": {"lr": 6e-4, "epochs": 20, "bs": 32, "max_len": 64},
    },
    "sst2": {
        "text_cols": ("sentence",),
        "num_labels": 2,
        "is_regression": False,
        "metric_for_best_model": "accuracy",
        "primary_metric": "accuracy",
        "defaults": {"lr": 6e-4, "epochs": 20, "bs": 32, "max_len": 64},
    },
    "mrpc": {
        "text_cols": ("sentence1", "sentence2"),
        "num_labels": 2,
        "is_regression": False,
        "metric_for_best_model": "f1",
        "primary_metric": "f1",
        "defaults": {"lr": 6e-4, "epochs": 20, "bs": 32, "max_len": 256},
    },
    "stsb": {
        "text_cols": ("sentence1", "sentence2"),
        "num_labels": 1,
        "is_regression": True,
        "metric_for_best_model": "combined_score",
        "primary_metric": "combined_score",
        "defaults": {"lr": 6e-4, "epochs": 20, "bs": 32, "max_len": 128},
    },
    "rte": {
        "text_cols": ("sentence1", "sentence2"),
        "num_labels": 2,
        "is_regression": False,
        "metric_for_best_model": "accuracy",
        "primary_metric": "accuracy",
        "defaults": {"lr": 6e-4, "epochs": 20, "bs": 32, "max_len": 256},
    },
    "qnli": {
        "text_cols": ("question", "sentence"),
        "num_labels": 2,
        "is_regression": False,
        "metric_for_best_model": "accuracy",
        "primary_metric": "accuracy",
        "defaults": {"lr": 6e-4, "epochs": 10, "bs": 32, "max_len": 128},
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_CONFIG.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name_or_path", default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--r", type=int, default=8, dest="rank")
    p.add_argument("--noise_alpha", type=float, default=0.01)
    p.add_argument("--noise_step_interval", type=int, default=5)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--per_device_eval_batch_size", type=int, default=None)
    p.add_argument("--max_seq_length", type=int, default=None)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--eval_mode",
        choices=["full_val", "psoft_val_test"],
        default="psoft_val_test",
        help="full_val: evaluate on the whole GLUE validation split. "
        "psoft_val_test: split validation 50/50 with current numpy RNG (seeded by --seed), "
        "use first half as val for best-checkpoint selection and report final metric on second half.",
    )
    p.add_argument(
        "--target_modules",
        nargs="+",
        default=["query_proj", "value_proj"],
        help="LoRA-family default = Q,V only. Pass 6 modules "
             "(query_proj key_proj value_proj attention.output.dense intermediate.dense output.dense) "
             "to match PSOFT's GLUE recipe.",
    )
    return p.parse_args()


def build_compute_metrics(task: str, is_regression: bool):
    metric = evaluate.load("glue", task)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if is_regression:
            preds = preds.squeeze(-1)
        else:
            preds = np.argmax(preds, axis=-1)
        result = metric.compute(predictions=preds, references=labels)
        if task == "stsb" and "pearson" in result and "spearmanr" in result:
            result["combined_score"] = (result["pearson"] + result["spearmanr"]) / 2
        return result

    return compute_metrics


def main() -> None:
    args = parse_args()
    cfg = TASK_CONFIG[args.task]
    defaults = cfg["defaults"]
    lr = args.lr if args.lr is not None else defaults["lr"]
    epochs = args.num_train_epochs if args.num_train_epochs is not None else defaults["epochs"]
    bs = args.per_device_train_batch_size if args.per_device_train_batch_size is not None else defaults["bs"]
    eval_bs = args.per_device_eval_batch_size if args.per_device_eval_batch_size is not None else bs
    max_len = args.max_seq_length if args.max_seq_length is not None else defaults["max_len"]

    set_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_datasets = load_dataset("glue", args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    text_cols = cfg["text_cols"]

    def tokenize_fn(examples):
        if len(text_cols) == 1:
            batch = tokenizer(examples[text_cols[0]], truncation=True, max_length=max_len)
        else:
            batch = tokenizer(
                examples[text_cols[0]],
                examples[text_cols[1]],
                truncation=True,
                max_length=max_len,
            )
        return batch

    tokenized = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in raw_datasets["train"].column_names if c not in ("label",)],
    ).rename_column("label", "labels")

    validation_split = tokenized["validation"]
    if args.eval_mode == "psoft_val_test":
        indices = np.arange(len(validation_split))
        np.random.shuffle(indices)
        half = len(indices) // 2
        val_dataset = validation_split.select(indices[:half].tolist())
        test_dataset = validation_split.select(indices[half:].tolist())
    else:
        val_dataset = validation_split
        test_dataset = validation_split

    model_kwargs: dict = {
        "num_labels": cfg["num_labels"],
        "torch_dtype": torch.float32,
    }
    if cfg["is_regression"]:
        model_kwargs["problem_type"] = "regression"
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, **model_kwargs)

    peft_config = CaraConfig(
        task_type="SEQ_CLS",
        r=args.rank,
        noise_alpha=args.noise_alpha,
        noise_step_interval=args.noise_step_interval,
        target_modules=args.target_modules,
        modules_to_save=["classifier", "pooler", "score"],
    )
    model = get_peft_model(model, peft_config)
    trainable_params, total_params, cara_params = 0, 0, 0
    for name, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
            if "cara_" in name:
                cara_params += p.numel()
    model.print_trainable_parameters()
    print(f"[cara_glue] cara-only params: {cara_params:,} ({cara_params / 1e6:.2f}M)")

    metric_key = cfg["metric_for_best_model"]
    training_args = TrainingArguments(
        output_dir=str(output_dir / "hf_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        logging_strategy="epoch",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_compute_metrics(args.task, cfg["is_regression"]),
    )

    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    primary_key_test = f"test_{cfg['primary_metric']}"
    primary_key_val = f"val_{cfg['primary_metric']}"
    primary_score = float(test_metrics.get(primary_key_test, val_metrics.get(primary_key_val)))

    result = {
        "task": args.task,
        "seed": args.seed,
        "model": args.model_name_or_path,
        "eval_mode": args.eval_mode,
        "r": args.rank,
        "target_modules": args.target_modules,
        "noise_alpha": args.noise_alpha,
        "noise_step_interval": args.noise_step_interval,
        "trainable_params": trainable_params,
        "cara_params": cara_params,
        "total_params": total_params,
        "primary_metric": cfg["primary_metric"],
        "primary_score": primary_score,
        "val_metrics": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "test_metrics": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
        "hyperparams": {
            "lr": lr,
            "epochs": epochs,
            "per_device_train_batch_size": bs,
            "per_device_eval_batch_size": eval_bs,
            "max_seq_length": max_len,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
        },
        "peft_config": asdict(peft_config),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[cara_glue] wrote {output_dir / 'result.json'}")
    print(f"[cara_glue] {args.task} seed={args.seed} {cfg['primary_metric']}={primary_score:.4f}")


if __name__ == "__main__":
    main()
