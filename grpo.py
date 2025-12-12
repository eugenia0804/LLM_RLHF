import torch
import torch.nn.functional as F

class OptimizedGRPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, optimizer, args, device):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.args = args
        self.device = device
        self.step_count = 0 

    def compute_group_advantages(self, rewards, group_size):
        # Normalize rewards within each group to compute advantages
        reshaped_rewards = rewards.view(-1, group_size)
        mean = reshaped_rewards.mean(dim=1, keepdim=True)
        std = reshaped_rewards.std(dim=1, keepdim=True)
        advantages = (reshaped_rewards - mean) / (std + 1e-8)
        return advantages.flatten()

    def compute_logprobs_and_entropy(self, logits, input_ids, mask):
        shift_logits = logits[:, :-1, :]
        shift_input_ids = input_ids[:, 1:]
        shift_mask = mask[:, 1:]

        log_softmax = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_softmax.gather(-1, shift_input_ids.unsqueeze(-1)).squeeze(-1)
        probs = torch.exp(log_softmax)
        entropy = -(probs * log_softmax).sum(dim=-1)
        
        return token_logprobs * shift_mask, entropy * shift_mask

    def grpo_step(self, batch):
        self.policy_model.train()
        
        gen_input_ids = batch["gen_input_ids"]
        response_mask = batch["response_mask"]
        old_logprobs = batch["old_logprobs"]
        advantages = batch["advantages"]
        
        # Forward Pass
        outputs = self.policy_model(
            input_ids=gen_input_ids, 
            attention_mask=batch["gen_attention_mask"],
            return_dict=True,
            compute_value=False 
        )
        logits = outputs.logits
        new_logprobs, entropy = self.compute_logprobs_and_entropy(logits, gen_input_ids, response_mask)
        
        # Ratio & Loss
        log_ratio = new_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        # clip ratio for stability
        ratio = torch.clip(ratio, 0.0, 10.0)
        
        adv_expanded = advantages.unsqueeze(-1)
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_ratio, 1.0 + self.args.clip_ratio) * adv_expanded
        
        loss_elementwise = -torch.min(surr1, surr2)
        response_mask_shifted = response_mask[:, 1:]

        valid_tokens = response_mask.sum()
        if valid_tokens == 0: valid_tokens = 1.0
        
        policy_loss = (loss_elementwise * response_mask_shifted).sum() / valid_tokens
        entropy_loss = -(entropy * response_mask_shifted).sum() / valid_tokens
        
        total_loss = policy_loss + (self.args.ent_coef * entropy_loss)
        
        # Gradient Accumulation
        loss_to_backward = total_loss / self.args.grad_acc_steps
        loss_to_backward.backward()
        
        self.step_count += 1
        
        if self.step_count % self.args.grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "loss/total": total_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/entropy": entropy_loss.item()
        }
