import argparse
import random
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import os
import matplotlib.pyplot as plt

# Load dataset and inspect a few examples for sanity
def load_and_inspect(dataset_name):
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)
    print("Dataset splits:", ds.keys())
    for split in ds.keys():
        print(f"\n--- sample from split {split} ---")
        for ex in ds[split].shuffle(seed=42).select(range(3)):
            sample = {k: ex.get(k) for k in ("chosen", "rejected")}
            print(sample, "\n")
    return ds

# Initialize tokenizer with padding and left truncation
def make_tokenizer(model_checkpoint):
    tok = AutoTokenizer.from_pretrained(model_checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    return tok

# Tokenize chosen/rejected pairs
def tokenize_batch(batch, tokenizer, max_length, store_text=False):
    chosen_texts, rejected_texts = batch["chosen"], batch["rejected"]

    tokens_chosen = tokenizer(chosen_texts, truncation=True, padding="max_length",
                              max_length=max_length, return_attention_mask=True)
    tokens_rejected = tokenizer(rejected_texts, truncation=True, padding="max_length",
                                max_length=max_length, return_attention_mask=True)

    out = {
        "chosen_input_ids": tokens_chosen["input_ids"],
        "chosen_attention_mask": tokens_chosen["attention_mask"],
        "chosen_length": [int(sum(x != tokenizer.pad_token_id for x in ids))
                          for ids in tokens_chosen["input_ids"]],
        "rejected_input_ids": tokens_rejected["input_ids"],
        "rejected_attention_mask": tokens_rejected["attention_mask"],
        "rejected_length": [int(sum(x != tokenizer.pad_token_id for x in ids))
                            for ids in tokens_rejected["input_ids"]],
    }

    if store_text:
        out["chosen_text"] = chosen_texts
        out["rejected_text"] = rejected_texts

    return out

# Filter extremely long examples
def rough_char_filter(example, max_chars=10000):
    return (example["chosen"] is not None and
            example["rejected"] is not None and
            len(example["chosen"]) <= max_chars and
            len(example["rejected"]) <= max_chars)

# Preprocess dataset: filter, split, tokenize, and save
def preprocess_dataset(raw_ds, tokenizer, max_length,
                       train_n=8000, val_n=2000, test_n=2000,
                       out_dir="processed_data", store_text=True):

    print("\nFiltering extremely long raw examples")
    filtered_train = raw_ds["train"].filter(rough_char_filter)
    filtered_test = raw_ds["test"].filter(rough_char_filter) if "test" in raw_ds else None

    # Shuffle and split train/val
    train_idxs = list(range(len(filtered_train)))
    random.shuffle(train_idxs)
    train_idxs = train_idxs[:train_n + val_n]
    train_split_idxs, val_split_idxs = train_idxs[:train_n], train_idxs[train_n:train_n+val_n]

    train_final, val_final = filtered_train.select(train_split_idxs), filtered_train.select(val_split_idxs)

    # Prepare test set
    test_final = None
    if filtered_test:
        test_idxs = list(range(len(filtered_test)))
        random.shuffle(test_idxs)
        test_final = filtered_test.select(test_idxs[:test_n])

    # Tokenize splits
    def tokenize_split(ds):
        return ds.map(lambda batch: tokenize_batch(batch, tokenizer, max_length, store_text),
                      batched=True, batch_size=1000, num_proc=4, remove_columns=ds.column_names)

    print("Tokenizing training set...")
    train_tok = tokenize_split(train_final)
    print("Tokenizing validation set...")
    val_tok = tokenize_split(val_final)
    test_tok = tokenize_split(test_final) if test_final else None
    if test_final:
        print("Tokenizing test set...")

    # Save final processed dataset
    final_ds = DatasetDict({"train": train_tok, "validation": val_tok})
    if test_tok:
        final_ds["test"] = test_tok

    os.makedirs(out_dir, exist_ok=True)
    final_ds.save_to_disk(out_dir)
    print(f"Saved processed datasets to {out_dir}")

    return final_ds

# Generate descriptive statistics and plots
def descriptive_analytics(ds, split="train", save_dir=None):
    print(f"\nAnalytics for {split}:")
    chosen, rejected = np.array(ds[split]["chosen_length"]), np.array(ds[split]["rejected_length"])
    n = len(chosen)

    print(f"Total number of pairs: {n}")
    print(f"Chosen lengths: mean={chosen.mean():.2f} | min={chosen.min()} | max={chosen.max()}")
    print(f"Rejected lengths: mean={rejected.mean():.2f} | min={rejected.min()} | max={rejected.max()}")

    plt.figure(figsize=(10,5))
    plt.hist(chosen, bins=50, alpha=0.6, label="Chosen")
    plt.hist(rejected, bins=50, alpha=0.6, label="Rejected")
    plt.legend()
    plt.title(f"Length Distribution – {split}")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{split}_hist.png")
        plt.savefig(out_path)
        print(f"Saved histogram to {out_path}")
    else:
        plt.show()

# Main pipeline: load, preprocess, and analyze dataset
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    p.add_argument("--model-checkpoint", type=str, default="gpt2")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--min-tokens", type=int, default=10)
    p.add_argument("--data-out-dir", type=str, default="data_reward")
    args = p.parse_args()

    raw = load_and_inspect(args.dataset)
    tokenizer = make_tokenizer(args.model_checkpoint)

    processed = preprocess_dataset(raw, tokenizer, args.max_length,
                                   train_n=10000, val_n=2000, test_n=2000,
                                   out_dir=args.data_out_dir)

    for split in processed.keys():
        descriptive_analytics(processed, split=split, save_dir=args.data_out_dir)

if __name__ == "__main__":
    main()
