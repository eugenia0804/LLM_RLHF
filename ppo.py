import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Collator for batching prompts with padding, truncation, and token sanitization
class PromptCollator:
    def __init__(self, pad_token_id, max_new_tokens, max_valid_id=50256):
        self.pad_token_id = pad_token_id
        self.max_valid_id = max_valid_id
        self.max_prompt_len = 1024 - max_new_tokens 

    def __call__(self, batch):
        # Process each prompt individually
        prompt_ids = []
        for x in batch:
            tensor = torch.tensor(x["prompt_input_ids"])
            tensor = tensor[tensor != self.pad_token_id]  # Remove existing padding
            tensor[tensor > self.max_valid_id] = self.pad_token_id  # Sanitize invalid tokens
            if len(tensor) > self.max_prompt_len:
                tensor = tensor[-self.max_prompt_len:]  # Truncate to max length
            prompt_ids.append(tensor)
            
        # Pad prompts to the same length
        max_len = max([x.size(0) for x in prompt_ids]) if prompt_ids else 1
        padded_input_ids = torch.full((len(prompt_ids), max_len), self.pad_token_id, dtype=torch.long)
        padded_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.long)
        
        for i, tensor in enumerate(prompt_ids):
            length = tensor.size(0)
            if length > 0:
                padded_input_ids[i, -length:] = tensor
                padded_mask[i, -length:] = 1
            
        return {
            "prompt_input_ids": padded_input_ids, 
            "prompt_attention_mask": padded_mask
        }

# Controller for adaptive KL coefficient
class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    # Update KL coefficient based on current KL divergence
    def update(self, current_kl, n_steps):
        error = current_kl - self.target
        adapt_rate = float(n_steps) / float(self.horizon)
        multiplier = 1.0 + np.clip(error / max(self.target, 1e-8), -0.2, 0.2) * adapt_rate
        self.value = float(np.clip(self.value * multiplier, 1e-6, 100.0))

# PPO trainer encapsulating loss computation and GAE
class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, optimizer, args, device):
        # Store models, optimizer, and arguments
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.args = args
        self.device = device
        self.kl_ctl = None
        if args.adaptive_kl:
            self.kl_ctl = AdaptiveKLController(args.kl_coef, args.target_kl, args.horizon)

    # Compute Generalized Advantage Estimation (GAE)
    def compute_gae(self, rewards, values, next_value, mask, input_ids, pad_id):
        device = rewards.device
        B, T = rewards.shape 
        
        advantages = torch.zeros_like(rewards, device=device, dtype=rewards.dtype)
        lastgaelam = torch.zeros(B, device=device, dtype=rewards.dtype)
        
        for t in reversed(range(T)):
            if t == T - 1:
                nextvalues = next_value
                nextnonterminal = (input_ids[:, t+1] != pad_id).float()
            else:
                nextvalues = values[:, t+1]
                nextnonterminal = (input_ids[:, t+1] != pad_id).float()

            delta = rewards[:, t] + self.args.gamma * nextvalues * nextnonterminal - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            advantages[:, t] = lastgaelam * mask[:, t].float()

        returns = advantages + values[:, :T]
        return advantages, returns

    # Compute log probabilities and entropy of actions
    def compute_logprobs_and_entropy(self, logits, input_ids, mask):
        # Shift logits for causal LM
        shift_logits = logits[:, :-1, :]
        shift_ids = input_ids[:, 1:]
        shift_mask = mask[:, 1:]
        
        log_softmax = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_softmax.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
        probs = torch.exp(log_softmax)
        entropy = -(probs * log_softmax).sum(dim=-1)
        
        # Apply mask
        token_logprobs = token_logprobs * shift_mask
        entropy = entropy * shift_mask
        
        return token_logprobs, entropy

    # Perform a single PPO step (loss computation only)
    def ppo_step(self, batch):
        self.policy_model.train()
        
        gen_input_ids = batch["gen_input_ids"]
        gen_mask = batch["gen_attention_mask"]
        response_mask = batch["response_mask"]
        old_logprobs = batch["old_logprobs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Shift mask to match logprobs length
        response_mask_shifted = response_mask[:, 1:]

        # Normalize advantages
        if self.args.normalize_advantages:
            valid_advs = advantages[response_mask_shifted.bool()]
            if valid_advs.numel() > 1:
                valid_advs = (valid_advs - valid_advs.mean()) / (valid_advs.std() + 1e-8)
                advantages[response_mask_shifted.bool()] = valid_advs
        
        # Forward pass
        outputs = self.policy_model(
            input_ids=gen_input_ids, 
            attention_mask=gen_mask, 
            return_dict=True,
            compute_value=True
        )
        logits = outputs.logits
        values = outputs.values.squeeze(-1)
        
        new_logprobs, entropy = self.compute_logprobs_and_entropy(logits, gen_input_ids, response_mask)
        
        # Compute PPO ratio
        log_ratio = new_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        ratio = torch.clamp(ratio, 0.0, 10.0)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_ratio, 1.0 + self.args.clip_ratio) * advantages
        loss_elementwise = -torch.min(surr1, surr2)
        
        valid_tokens = response_mask_shifted.sum()
        if valid_tokens == 0: valid_tokens = 1.0
        
        policy_loss = (loss_elementwise * response_mask_shifted).sum() / valid_tokens
        
        # Value loss
        values_shifted = values[:, :-1]
        value_loss = 0.5 * ((values_shifted - returns)**2 * response_mask_shifted).sum() / valid_tokens

        # Entropy loss
        entropy_loss = -(entropy * response_mask_shifted).sum() / valid_tokens
        
        # Total loss
        total_loss = policy_loss + (self.args.value_coef * value_loss) + (self.args.ent_coef * entropy_loss)
        
        # Return loss and metrics without backward
        return total_loss, {
            "loss/total": total_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "loss/entropy": entropy_loss.item()
        }
