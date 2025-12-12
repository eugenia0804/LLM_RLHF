import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from datasets import load_from_disk
from tqdm.auto import tqdm
import torch.nn.functional as F

# Assumes you have these files
from model_policy import PolicyModel
from model_reward import RewardModel
from vis import plot_train_loss, plot_rlhf_stats
from grpo import OptimizedGRPOTrainer

device = torch.device("cuda:1")


class LeftPadPromptCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Convert raw lists to tensors
        input_ids = [torch.tensor(x["prompt_input_ids"]) for x in batch]
        masks = [torch.tensor(x["prompt_attention_mask"]) for x in batch]
        
        # Left-pad everything to the longest prompt in batch
        max_len = max([x.size(0) for x in input_ids])
        
        padded_ids = []
        padded_masks = []
        
        for id_t, mask_t in zip(input_ids, masks):
            pad_len = max_len - id_t.size(0)
            if pad_len > 0:
                # Left padding: prepend pad tokens + zero mask
                pad_tensor = torch.full((pad_len,), self.pad_token_id, dtype=id_t.dtype)
                mask_pad_tensor = torch.zeros((pad_len,), dtype=mask_t.dtype)
                padded_ids.append(torch.cat([pad_tensor, id_t]))
                padded_masks.append(torch.cat([mask_pad_tensor, mask_t]))
            else:
                padded_ids.append(id_t)
                padded_masks.append(mask_t)
        
        # Return padded batch (B, T)
        return {
            "prompt_input_ids": torch.stack(padded_ids),
            "prompt_attention_mask": torch.stack(padded_masks)
        }

def train(args):
    print(f"--- Starting Optimized GRPO Training on {device} ---")

    # 1. Load Tokenizer (left padding for autoregressive generation)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id 

    # 2. Load Dataset + collator that performs left padding
    ds = load_from_disk(args.data_dir)
    collator = LeftPadPromptCollator(pad_token_id=pad_id)
    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # 3. Initialize models (policy + ref for KL + reward model)
    print(f"Loading Policy: {args.policy_checkpoint}")
    policy_model = PolicyModel(args.policy_checkpoint).to(device)
    ref_model = PolicyModel(args.policy_checkpoint).to(device).eval().requires_grad_(False)  # frozen baseline
    
    print("Loading Reward Model...")
    reward_model = RewardModel("gpt2").to(device)
    
    # Remap keys for custom RM architecture
    state_dict = torch.load(args.reward_checkpoint, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("lm.transformer", "backbone").replace("reward_head", "head").replace("lm.", "")
        new_state_dict[new_key] = v
    reward_model.load_state_dict(new_state_dict, strict=False)
    reward_model.eval().requires_grad_(False)

    # Optimizer + scheduler (cosine decay)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs
    total_updates = total_steps // args.grad_acc_steps  # scheduler steps on update, not fwd pass
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=20, num_training_steps=total_updates)

    trainer = OptimizedGRPOTrainer(policy_model, ref_model, reward_model, optimizer, args, device)

    # Main GRPO training loop
    global_step = 0
    history = {"steps": [], "loss": [], "reward": [], "kl": []}
    raw_losses = []
    best_loss = float("inf")   # track best rolling window

    optimizer.zero_grad() 

    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            global_step += 1
            
            # Prompt inputs -> device
            prompts = batch["prompt_input_ids"].to(device)
            prompt_masks = batch["prompt_attention_mask"].to(device)
            
            # GRPO: repeat each prompt G times (group sampling)
            prompts = prompts.repeat_interleave(args.group_size, dim=0)
            prompt_masks = prompt_masks.repeat_interleave(args.group_size, dim=0)

            # Autoregressive generation under policy
            with torch.no_grad():
                gen_seqs = policy_model.generate(
                    input_ids=prompts, attention_mask=prompt_masks,
                    max_new_tokens=args.max_new_tokens, do_sample=True, top_k=args.top_k,
                    pad_token_id=pad_id 
                )
            
            # Compute logprobs, KL, and reward
            with torch.no_grad():
                # Build response mask to isolate generated tokens
                prompt_lens = prompt_masks.sum(dim=1)
                response_mask = torch.zeros_like(gen_seqs)
                for i, L in enumerate(prompt_lens):
                    response_mask[i, L:] = 1
                
                # Logprobs under ref + policy
                ref_logprobs, _ = trainer.compute_logprobs_and_entropy(ref_model(gen_seqs).logits, gen_seqs, response_mask)
                old_logprobs, _ = trainer.compute_logprobs_and_entropy(policy_model(gen_seqs, compute_value=False).logits, gen_seqs, response_mask)
                
                # KL per token
                kl_div = old_logprobs - ref_logprobs
                response_mask_shifted = response_mask[:, 1:]
                
                # Logging KL (unclamped)
                current_kl = (kl_div * response_mask_shifted).sum() / (response_mask.sum() + 1e-8)
                
                # Reward KL uses only positive deviation (avoid negative KL)
                kl_penalty = torch.clamp(kl_div, min=0.0)
                
                rewards_dense = -args.kl_coef * kl_penalty * response_mask_shifted
                
                # Reward model score per sequence (clamped for stability)
                reward_inputs = gen_seqs.clone()
                reward_inputs[reward_inputs >= 50257] = pad_id 
                raw_rewards = reward_model(reward_inputs, attention_mask=(reward_inputs!=pad_id)).squeeze(-1)
                raw_rewards = torch.clamp(raw_rewards, min=-3.0, max=3.0)  # avoid exploding rewards
                
                # Normalize sequence reward across response length
                valid_len = response_mask_shifted.sum(dim=1, keepdim=True) + 1e-8
                rewards_dense += raw_rewards[:, None] / valid_len

                # GRPO: form group-level advantages
                total_seq_rewards = rewards_dense.sum(dim=1)
                advantages = trainer.compute_group_advantages(total_seq_rewards, args.group_size)

            # One GRPO update step
            batch_data = {
                "gen_input_ids": gen_seqs, 
                "gen_attention_mask": (gen_seqs!=pad_id).long(),
                "response_mask": response_mask, 
                "old_logprobs": old_logprobs,
                "advantages": advantages
            }
            
            metrics = trainer.grpo_step(batch_data)
            
            # Scheduler step only after a full update
            if trainer.step_count % args.grad_acc_steps == 0:
                lr_scheduler.step()
            
            # Logging metrics
            mean_reward = raw_rewards.mean().item()
            raw_losses.append(metrics["loss/total"])

            # Save best rolling model every 40 steps
            if global_step % 40 == 0:
                recent_losses = raw_losses[-40:]
                avg_recent = sum(recent_losses) / len(recent_losses)

                if avg_recent < best_loss:
                    best_loss = avg_recent
                    os.makedirs(args.output_dir, exist_ok=True)
                    save_path = os.path.join(args.output_dir, f"best_policy_step{global_step}.pt")
                    torch.save(policy_model.state_dict(), save_path)
                    print(f"[Step {global_step}] New best model saved. AvgLoss={avg_recent:.4f}")

            
            # Logging window for plots
            if global_step % args.log_steps == 0:
                avg_loss = sum(raw_losses[-args.log_steps:]) / len(raw_losses[-args.log_steps:])
                history["steps"].append(global_step)
                history["loss"].append(avg_loss)
                history["reward"].append(mean_reward)
                history["kl"].append(current_kl.item())
                
                plot_train_loss(history["steps"], history["loss"], args.output_dir, log_scale=True)
                plot_rlhf_stats(history["steps"], history["reward"], history["kl"], args.output_dir)
                
                metrics["kl"] = round(current_kl.item(), 4)
                metrics["rew"] = round(mean_reward, 4)
                progress_bar.set_postfix(metrics)
                
            # Cleanup to reduce VRAM spikes
            del gen_seqs, prompts, prompt_masks, batch_data, advantages
            torch.cuda.empty_cache()

    # Final save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(policy_model.state_dict(), os.path.join(args.output_dir, "final_grpo_policy.pt"))
    print("Optimization Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_rlhf")
    parser.add_argument("--policy-checkpoint", type=str, default="Dahoas/gpt2-sft-static") 
    parser.add_argument("--reward-checkpoint", type=str, default="results_reward/best_reward_model.pt")
    parser.add_argument("--output-dir", type=str, default="results_grpo_opt")

    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-acc-steps", type=int, default=2)
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6) 
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=20)

    args = parser.parse_args()
    train(args)