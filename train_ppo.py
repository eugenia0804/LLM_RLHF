import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from datasets import load_from_disk
from tqdm.auto import tqdm

from model_policy import PolicyModel
from model_reward import RewardModel
from ppo import PPOTrainer, PromptCollator
from vis import plot_train_loss, plot_rlhf_stats

device = torch.device("cuda:4")

def train(args):
    print(f"--- Starting Training with PPO on {device} (Mixed Precision) ---")

    # Load Tokenizer & Data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id  # For convenience in masking/padding ops

    ds = load_from_disk(args.data_dir)
    collator = PromptCollator(
        pad_token_id=pad_id,
        max_new_tokens=args.max_new_tokens,
        max_valid_id=50256  # upper boundary for valid indices
    )
    train_loader = DataLoader(
        ds["train"], batch_size=args.batch_size,
        shuffle=True, collate_fn=collator
    )

    # Load Models
    print(f"Initializing Policy from: {args.policy_checkpoint}")
    policy_model = PolicyModel(args.policy_checkpoint).to(device)
    ref_model = PolicyModel(args.policy_checkpoint).to(device)
    ref_model.eval()  # ref model is fixed baseline
    
    reward_model = RewardModel("gpt2").to(device)
    state_dict = torch.load(args.reward_checkpoint, map_location=device)

    # Convert reward model checkpoints to internal naming scheme
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = (
            k.replace("lm.transformer", "backbone")
             .replace("reward_head", "head")
             .replace("lm.", "")
        )
        new_state_dict[new_key] = v
    reward_model.load_state_dict(new_state_dict, strict=False)
    reward_model.eval().requires_grad_(False)  # Reward model is frozen

    # Optimizer & Scaler
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")  # Mixed precision scaler

    total_steps = len(train_loader) * args.epochs
    update_steps = total_steps // args.grad_acc_steps  # scheduler sees *actual* update steps
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=update_steps
    )

    trainer = PPOTrainer(policy_model, ref_model, reward_model, optimizer, args, device)
    
    # Training Loop
    global_step = 0
    history = {"steps": [], "loss": [], "reward": [], "kl": []}
    step_metrics = {"loss": [], "reward": [], "kl": []}
    
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for i, batch in enumerate(progress_bar):
            prompts = batch["prompt_input_ids"].to(device)
            prompt_masks = batch["prompt_attention_mask"].to(device)
            
            # Phase 1: Generate responses and compute old policy logprobs/value
            with torch.no_grad():
                gen_seqs = policy_model.generate(
                    input_ids=prompts, attention_mask=prompt_masks,
                    max_new_tokens=args.max_new_tokens, do_sample=True,
                    top_k=args.top_k, pad_token_id=pad_id
                )
                
                # Build response mask (0 on prompt tokens, 1 on response tokens)
                prompt_lens = prompt_masks.sum(dim=1)
                response_mask = torch.zeros_like(gen_seqs)
                for idx, L in enumerate(prompt_lens):
                    response_mask[idx, L:] = 1
                
                # Logprobs under reference policy (baseline)
                ref_logprobs, _ = trainer.compute_logprobs_and_entropy(
                    ref_model(gen_seqs).logits, gen_seqs, response_mask
                )
                
                # Logprobs + values from *current* policy
                old_outputs = policy_model(gen_seqs, compute_value=True)
                old_logprobs, _ = trainer.compute_logprobs_and_entropy(
                    old_outputs.logits, gen_seqs, response_mask
                )
                old_values = old_outputs.values.squeeze(-1)
                
                # Shift mask to align with next-token logprobs shapes
                response_mask_shifted = response_mask[:, 1:]

                # KL Divergence (token-level)
                kl_div = old_logprobs - ref_logprobs
                current_kl = (
                    (kl_div * response_mask_shifted).sum()
                    / (response_mask_shifted.sum() + 1e-8)
                )
                
                # KL reward shaping
                kl_beta = trainer.kl_ctl.value if trainer.kl_ctl else args.kl_coef
                rewards_dense = -kl_beta * kl_div * response_mask_shifted
                
                # Sequence-level reward from reward model
                r_in = gen_seqs.clone()
                r_in[r_in >= 50257] = pad_id  # clamp OOV tokens
                raw_rewards = reward_model(
                    r_in, attention_mask=(r_in != pad_id)
                ).squeeze(-1)
                
                # Spread sequence reward across valid response positions
                rewards_dense += raw_rewards[:, None] / (
                    response_mask_shifted.sum(dim=1, keepdim=True) + 1e-8
                )

                # Compute bootstrapped value for last valid token
                B_sz = gen_seqs.size(0)
                last_nonpad = (gen_seqs != pad_id).sum(dim=1) - 1
                last_nonpad = last_nonpad.clamp(min=0)
                next_val = old_values[
                    torch.arange(B_sz, device=device), last_nonpad
                ]

                # GAE advantage estimation
                advantages, returns = trainer.compute_gae(
                    rewards=rewards_dense,
                    values=old_values,
                    next_value=next_val,
                    mask=response_mask_shifted,
                    input_ids=gen_seqs,
                    pad_id=pad_id
                )
                
                # Normalize only on valid positions
                adv_valid = advantages[response_mask_shifted.bool()]
                if adv_valid.numel() > 1:
                    advantages[response_mask_shifted.bool()] = (
                        (adv_valid - adv_valid.mean())
                        / (adv_valid.std() + 1e-8)
                    )

            # Phase 2: PPO update
            batch_data = {
                "gen_input_ids": gen_seqs,
                "gen_attention_mask": (gen_seqs != pad_id).long(),
                "response_mask": response_mask,
                "old_logprobs": old_logprobs,
                "advantages": advantages,
                "returns": returns
            }
            
            # Mixed-precision forward
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss, metrics = trainer.ppo_step(batch_data)
                loss = loss / args.grad_acc_steps  # gradient accumulation scaling

            # Backward pass (scaled)
            scaler.scale(loss).backward()
            
            # Track running statistics
            step_metrics["loss"].append(metrics["loss/total"])
            step_metrics["reward"].append(raw_rewards.mean().item())
            step_metrics["kl"].append(current_kl.item())
            
            # Update after enough accumulation steps
            if (i + 1) % args.grad_acc_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(),
                    args.max_grad_norm
                )

                scaler.step(optimizer)
                scaler.update()
                
                lr_scheduler.step()  # scheduler progresses per update
                optimizer.zero_grad()
                
                if trainer.kl_ctl:
                    # Update adaptive KL controller
                    trainer.kl_ctl.update(
                        current_kl.item(),
                        args.batch_size * args.grad_acc_steps
                    )
                
                global_step += 1
                
                # Logging & plot updates
                if global_step % args.log_steps == 0:
                    avg_loss = sum(step_metrics["loss"]) / len(step_metrics["loss"])
                    avg_rew = sum(step_metrics["reward"]) / len(step_metrics["reward"])
                    avg_kl = sum(step_metrics["kl"]) / len(step_metrics["kl"])
                    
                    history["steps"].append(global_step)
                    history["loss"].append(avg_loss)
                    history["reward"].append(avg_rew)
                    history["kl"].append(avg_kl)
                    
                    plot_train_loss(history["steps"], history["loss"], args.output_dir, log_scale=False)
                    plot_rlhf_stats(history["steps"], history["reward"], history["kl"], args.output_dir)
                    
                    print(f"\n[Step {global_step}] Loss: {avg_loss:.4f} | Reward: {avg_rew:.4f} | KL: {avg_kl:.4f}")
                    step_metrics = {k: [] for k in step_metrics}
            
            # Explicit cleanup to reduce VRAM fragmentation
            del gen_seqs, prompts, prompt_masks, batch_data, advantages, returns
            del old_logprobs, ref_logprobs, rewards_dense, old_values
            torch.cuda.empty_cache()

    # Save final policy checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(policy_model.state_dict(), os.path.join(args.output_dir, "final_policy.pt"))
    print("PPO Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_rlhf")
    parser.add_argument("--policy-checkpoint", type=str, default="Dahoas/gpt2-sft-static") 
    parser.add_argument("--reward-checkpoint", type=str, default="results_reward/final_reward_model.pt")
    parser.add_argument("--output-dir", type=str, default="results_ppo")
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-acc-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6) 
    parser.add_argument("--kl-coef", type=float, default=0.05)
    
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--normalize-advantages", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--adaptive-kl", action="store_true", default=True)
    parser.add_argument("--target-kl", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=10000)
    parser.add_argument("--log-steps", type=int, default=5)

    args = parser.parse_args()
    train(args)
