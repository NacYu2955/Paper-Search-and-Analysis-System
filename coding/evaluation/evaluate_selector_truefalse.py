import argparse
import csv
import gc
import json
import os
import re
import time
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LABEL_PATTERN = re.compile(r"\b(True|False)\b", re.IGNORECASE)


def parse_label(text: str) -> Optional[bool]:
    match = LABEL_PATTERN.search(text or "")
    if not match:
        return None
    return match.group(1).lower() == "true"


def load_testset(path: str, max_samples: Optional[int] = None) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if len(messages) < 2:
                raise ValueError(f"Line {line_number} does not contain user/assistant messages")
            prompt = messages[0].get("content", "")
            gold_text = messages[1].get("content", "")
            gold = parse_label(gold_text)
            if gold is None:
                raise ValueError(f"Line {line_number} has no True/False gold label: {gold_text!r}")
            rows.append(
                {
                    "line_number": line_number,
                    "prompt": prompt,
                    "gold": gold,
                    "gold_text": gold_text,
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def batched(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return tokenizer, model


def render_prompts(tokenizer, prompts: List[str]) -> List[str]:
    rendered = []
    for prompt in prompts:
        rendered.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return rendered


def predict_batch(
    tokenizer,
    model,
    batch: List[dict],
    *,
    max_input_length: int,
    max_new_tokens: int,
) -> List[dict]:
    rendered = render_prompts(tokenizer, [item["prompt"] for item in batch])
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = encoded["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)

    outputs = []
    for item, text in zip(batch, decoded):
        pred = parse_label(text)
        outputs.append(
            {
                **item,
                "prediction": pred,
                "prediction_text": text.strip(),
            }
        )

    del encoded, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def compute_metrics(results: List[dict]) -> Dict[str, float]:
    total = len(results)
    invalid = sum(1 for row in results if row["prediction"] is None)

    tp = tn = fp = fn = 0
    correct = 0
    for row in results:
        gold = row["gold"]
        pred = row["prediction"]

        # Invalid generations are counted as wrong and mapped to the opposite
        # class for binary confusion metrics.
        if pred is None:
            pred = not gold

        if pred == gold:
            correct += 1
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = correct / total if total else 0.0

    return {
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "invalid": invalid,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_model(
    *,
    model_name: str,
    model_path: str,
    records: List[dict],
    output_dir: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    print(f"\n=== Evaluating {model_name} ===")
    print(f"Model path: {model_path}")
    tokenizer, model = load_model(model_path)

    details = []
    start_time = time.perf_counter()
    for batch_id, batch in enumerate(batched(records, batch_size), start=1):
        details.extend(
            predict_batch(
                tokenizer,
                model,
                batch,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
            )
        )
        print(f"Processed {len(details)}/{len(records)} samples", flush=True)

    elapsed = time.perf_counter() - start_time
    metrics = compute_metrics(details)
    metrics["model_name"] = model_name
    metrics["model_path"] = model_path
    metrics["seconds"] = elapsed
    metrics["seconds_per_sample"] = elapsed / len(records) if records else 0.0

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    detail_path = os.path.join(output_dir, f"{safe_name}_truefalse_details.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "line_number",
                "gold",
                "prediction",
                "prediction_text",
                "gold_text",
                "prompt",
            ],
        )
        writer.writeheader()
        for row in details:
            writer.writerow(
                {
                    "line_number": row["line_number"],
                    "gold": "True" if row["gold"] else "False",
                    "prediction": (
                        ""
                        if row["prediction"] is None
                        else ("True" if row["prediction"] else "False")
                    ),
                    "prediction_text": row["prediction_text"],
                    "gold_text": row["gold_text"],
                    "prompt": row["prompt"],
                }
            )

    print(
        "Accuracy={accuracy:.4f} Precision={precision:.4f} Recall={recall:.4f} "
        "F1={f1:.4f} Invalid={invalid}".format(**metrics)
    )
    print(f"Details: {detail_path}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def write_summary(output_dir: str, metrics_rows: List[Dict[str, float]]) -> None:
    summary_csv = os.path.join(output_dir, "selector_truefalse_summary.csv")
    summary_json = os.path.join(output_dir, "selector_truefalse_summary.json")
    fieldnames = [
        "model_name",
        "model_path",
        "total",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "invalid",
        "tp",
        "tn",
        "fp",
        "fn",
        "seconds",
        "seconds_per_sample",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(metrics_rows, file, indent=2, ensure_ascii=False)
    print(f"\nSummary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base Qwen and fine-tuned selector on SFT True/False test data."
    )
    parser.add_argument("--dataset", default=os.path.join("dataset", "sft", "test.jsonl"))
    parser.add_argument("--base-model", required=True, help="Path to original Qwen2.5-7B-Instruct")
    parser.add_argument("--ft-model", required=True, help="Path to fine-tuned selector model")
    parser.add_argument("--output-dir", default=os.path.join("output", "selector_eval"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_testset(args.dataset, max_samples=args.max_samples)
    positives = sum(1 for row in records if row["gold"])
    negatives = len(records) - positives
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(records)} | True: {positives} | False: {negatives}")

    metrics_rows = []
    metrics_rows.append(
        evaluate_model(
            model_name="base_qwen",
            model_path=args.base_model,
            records=records,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
        )
    )
    metrics_rows.append(
        evaluate_model(
            model_name="fine_tuned_selector",
            model_path=args.ft_model,
            records=records,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
        )
    )
    write_summary(args.output_dir, metrics_rows)


if __name__ == "__main__":
    main()
