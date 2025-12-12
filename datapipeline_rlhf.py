import argparse
import random
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import os
import matplotlib.pyplot as plt

# Split text into prompt and response based on last 'Assistant:' occurrence
def split_prompt_response(text):
    marker = "Assistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return text, ""
    prompt = text[:idx].strip()
    response = text[idx + len(marker):].strip()
    return prompt, response

# Load dataset and inspect a few examples for sanity check
def load_and_inspect(dataset_name):
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)
    print("Dataset splits:", ds.keys())
    for split in ds.keys():
        print(f"\n--- sample from split {split} ---")
        for ex in ds[split].shuffle(seed=42).select(range(3)):
            prompt_c, resp_c = split_prompt_response(ex["chosen"])
            prompt_r, resp_r = split_prompt_response(ex["rejected"])
            print({
                "prompt_chosen": prompt_c[:200],
                "response_chosen": resp_c[:200],
                "prompt_rejected": prompt_r[:200],
                "response_rejected": resp_r[:200],
            }, "\n")
    return ds

# Initialize tokenizer with pad token
def make_tokenizer(model_checkpoint):
    tok = AutoTokenizer.from_pretrained(model_checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# Tokenize dataset into PPO-ready inputs
def tokenize_for_ppo(batch, tokenizer, max_length, store_text=False):
    prompts, responses = [], []
    for c in batch["chosen"]:
        p, r = split_prompt_response(c)
        prompts.append(p)
        responses.append(r)

    token_prompt = tokenizer(prompts, truncation=True, padding=False,
                             max_length=max_length, return_attention_mask=True)
    token_response = tokenizer(responses, truncation=True, padding=False,
                               max_length=max_length, return_attention_mask=True)

    out = {
        "prompt_input_ids": token_prompt["input_ids"],
        "prompt_attention_mask": token_prompt["attention_mask"],
        "prompt_length": [int(sum(t != tokenizer.pad_token_id for t in ids))
                          for ids in token_prompt["input_ids"]],
        "response_input_ids": token_response["input_ids"],
        "response_attention_mask": token_response["attention_mask"],
        "response_length": [int(sum(t != tokenizer.pad_token_id for t in ids))
                            for ids in token_response["input_ids"]],
    }

    if store_text:
        out["prompt_text"] = prompts
        out["response_text"] = responses

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
                       out_dir="ppo_data", store_text=True):

    print("\nFiltering extremely long raw examples")
    filtered_train = raw_ds["train"].filter(rough_char_filter)
    filtered_test = raw_ds["test"].filter(rough_char_filter) if "test" in raw_ds else None

    # Shuffle and select subsets for train/val/test
    idxs = list(range(len(filtered_train)))
    random.shuffle(idxs)
    idxs = idxs[:train_n + val_n]
    train_split, val_split = idxs[:train_n], idxs[train_n:train_n + val_n]

    train_final, val_final = filtered_train.select(train_split), filtered_train.select(val_split)
    test_final = None
    if filtered_test:
        tidx = list(range(len(filtered_test)))
        random.shuffle(tidx)
        test_final = filtered_test.select(tidx[:test_n])

    # Tokenize function
    def t(ds):
        return ds.map(lambda batch: tokenize_for_ppo(batch, tokenizer, max_length, store_text),
                      batched=True, batch_size=1000, num_proc=4, remove_columns=ds.column_names)

    print("Tokenizing training set...")
    train_tok = t(train_final)
    print("Tokenizing validation set...")
    val_tok = t(val_final)
    test_tok = t(test_final) if test_final else None

    # Save final PPO-formatted dataset
    final_ds = DatasetDict({"train": train_tok, "validation": val_tok})
    if test_tok:
        final_ds["test"] = test_tok

    os.makedirs(out_dir, exist_ok=True)
    final_ds.save_to_disk(out_dir)
    print(f"Saved PPO-formatted datasets to {out_dir}")
    return final_ds

# Generate basic dataset statistics and histograms
def descriptive_analytics(ds, split="train", save_dir=None):
    print(f"\nAnalytics for {split}:")
    p, r = np.array(ds[split]["prompt_length"]), np.array(ds[split]["response_length"])
    print(f"* Prompt length – mean:{p.mean():.2f}  min:{p.min()}  max:{p.max()}")
    print(f"* Response length – mean:{r.mean():.2f}  min:{r.min()}  max:{r.max()}")

    plt.figure(figsize=(10, 5))
    plt.hist(p, bins=50, alpha=0.6, label="Prompt")
    plt.hist(r, bins=50, alpha=0.6, label="Response")
    plt.legend()
    plt.title(f"Token Length Distribution – {split}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{split}_hist.png")
        plt.savefig(out)
        print(f"Saved histogram to {out}")
    else:
        plt.show()

# Main pipeline: load, preprocess, and analyze dataset
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    p.add_argument("--model-checkpoint", type=str, default="gpt2")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--data-out-dir", type=str, default="data_rlhf")
    args = p.parse_args()

    raw = load_and_inspect(args.dataset)
    tokenizer = make_tokenizer(args.model_checkpoint)

    processed = preprocess_dataset(raw, tokenizer, args.max_length,
                                   train_n=10000, val_n=2000, test_n=2000,
                                   out_dir=args.data_out_dir, store_text=True)

    for split in processed.keys():
        descriptive_analytics(processed, split=split, save_dir=args.data_out_dir)

if __name__ == "__main__":
    main()
