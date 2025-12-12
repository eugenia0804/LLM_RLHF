import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from model_policy import PolicyModel
from dpo import DPOTrainer
from vis import plot_train_loss

device = torch.device("cuda:2")

def split_prompt_response(text):
    """
    Splits Anthropic format: "Human: ... \n\nAssistant: ..."
    Returns (prompt, response)
    """
    marker = "Assistant:"
    idx = text.rfind(marker)
    if idx == -1: return text, ""
    prompt = text[:idx + len(marker)].strip()
    response = text[idx + len(marker):].strip()
    return prompt, response

class DPOCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        prompts = []
        chosen_resps = []
        rejected_resps = []
        
        for item in batch:
            p, c = split_prompt_response(item["chosen"])
            _, r = split_prompt_response(item["rejected"])
            prompts.append(p)
            chosen_resps.append(c)
            rejected_resps.append(r)
            
        def tokenize_pair(p_list, r_list):
            full_texts = [p + " " + r for p, r in zip(p_list, r_list)]
            enc = self.tokenizer(
                full_texts, padding=True, truncation=True, 
                max_length=self.max_length, return_tensors="pt"
            )
            labels = enc["input_ids"].clone()
            
            # Mask out the prompt part
            p_enc = self.tokenizer(p_list, padding=False, truncation=True, max_length=self.max_length)
            
            for i, p_ids in enumerate(p_enc["input_ids"]):
                L = len(p_ids)
                labels[i, :L] = -100
                labels[i][enc["attention_mask"][i] == 0] = -100
                
            return enc["input_ids"], enc["attention_mask"], labels

        c_ids, c_mask, c_labels = tokenize_pair(prompts, chosen_resps)
        r_ids, r_mask, r_labels = tokenize_pair(prompts, rejected_resps)
        
        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "chosen_labels": c_labels,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
            "rejected_labels": r_labels
        }

def plot_metrics(history, output_dir):
    """Helper to plot additional metrics beyond just loss"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Accuracy
    plt.figure()
    plt.plot(history["steps"], history["acc"], label="Reward Accuracy")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "train_accuracy.png"))
    plt.close()

    # Plot Margin
    plt.figure()
    plt.plot(history["steps"], history["margin"], label="Reward Margin", color="orange")
    plt.xlabel("Steps")
    plt.ylabel("Margin (Chosen - Rejected)")
    plt.title("Reward Margin")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "train_margin.png"))
    plt.close()

def train(args):
    print(f"--- Starting DPO Training on {device} ---")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Dataset
    print(f"Loading dataset: {args.dataset_name}...")
    # Loading a subset for demonstration/efficiency as per assignment recommendation
    ds = load_dataset(args.dataset_name, split="train[:2000]") 
    
    collator = DPOCollator(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # Initialize Models
    print(f"Initializing Policy from: {args.policy_checkpoint}")
    policy_model = PolicyModel(args.policy_checkpoint).to(device)
    
    print("Initializing Reference Model...")
    ref_model = PolicyModel(args.policy_checkpoint).to(device)
    ref_model.eval()

    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr)
    
    # Adjust scheduler for gradient accumulation
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=50, num_training_steps=max_train_steps)

    trainer = DPOTrainer(policy_model, ref_model, optimizer, args, device)

    # Logging buffers
    history = {
        "steps": [], 
        "loss": [], 
        "acc": [], 
        "margin": [],
        "reward_chosen": [],
        "reward_rejected": []
    }
    
    # Temporary buffer for smoothing
    step_metrics = {"loss": [], "acc": [], "margin": [], "chosen": [], "rejected": []}
    
    global_step = 0
    
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for i, batch in enumerate(progress_bar):
            # 1. DPO Step (Get Tensor AND Metrics)
            loss, metrics = trainer.dpo_step(batch)  # <--- CHANGED
            
            # Normalize loss for accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward() # <--- Now works because 'loss' is attached to the graph
            
            # Store raw metrics
            step_metrics["loss"].append(metrics["loss/dpo"])
            step_metrics["acc"].append(metrics["reward/accuracy"])
            step_metrics["margin"].append(metrics["reward/margin"])
            step_metrics["chosen"].append(metrics["reward/chosen"])
            step_metrics["rejected"].append(metrics["reward/rejected"])
            
            # 2. Optimizer Step (Gradient Accumulation)
            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 3. Logging (Only log on update steps)
                if global_step % args.log_steps == 0:
                    # Average metrics over the accumulation window
                    avg_loss = sum(step_metrics["loss"]) / len(step_metrics["loss"])
                    avg_acc = sum(step_metrics["acc"]) / len(step_metrics["acc"])
                    avg_margin = sum(step_metrics["margin"]) / len(step_metrics["margin"])
                    avg_chosen = sum(step_metrics["chosen"]) / len(step_metrics["chosen"])
                    avg_rejected = sum(step_metrics["rejected"]) / len(step_metrics["rejected"])
                    
                    history["steps"].append(global_step)
                    history["loss"].append(avg_loss)
                    history["acc"].append(avg_acc)
                    history["margin"].append(avg_margin)
                    history["reward_chosen"].append(avg_chosen)
                    history["reward_rejected"].append(avg_rejected)
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "acc": f"{avg_acc:.2f}",
                        "margin": f"{avg_margin:.3f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                    # Update plots
                    plot_train_loss(history["steps"], history["loss"], args.output_dir)
                    plot_metrics(history, args.output_dir)
                    
                    # Clear buffers
                    step_metrics = {k: [] for k in step_metrics}

    # Save Final Model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(policy_model.state_dict(), os.path.join(args.output_dir, "final_dpo_policy.pt"))
    print(f"Training Complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--policy-checkpoint", type=str, default="Dahoas/gpt2-sft-static")
    parser.add_argument("--output-dir", type=str, default="results_dpo_opt")
    
    parser.add_argument("--epochs", type=int, default=3)
    
    # OPTIMIZED DEFAULTS for Stability
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per forward pass")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Steps to accumulate before update (Effective Batch = 64)")
    
    parser.add_argument("--lr", type=float, default=1e-6, help="Lower LR for stability")
    parser.add_argument("--beta", type=float, default=0.1, help="Lower Beta (0.1) prevents reward hacking/instability")
    
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=1)

    args = parser.parse_args()
    
    # Ensure divisible
    if args.gradient_accumulation_steps < 1:
        args.gradient_accumulation_steps = 1
        
    train(args)