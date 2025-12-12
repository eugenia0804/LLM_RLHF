import torch
import torch.nn.functional as F

class DPOTrainer:
    def __init__(self, policy_model, ref_model, optimizer, args, device):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.args = args
        self.device = device
        
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

    def _get_batch_logps(self, logits, labels, average_log_prob=False):
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same batch and sequence length dimensions.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)
        labels[labels == -100] = 0 

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def dpo_step(self, batch):
        self.policy_model.train()
        
        pos_ids = batch["chosen_input_ids"].to(self.device)
        pos_mask = batch["chosen_attention_mask"].to(self.device)
        pos_labels = batch["chosen_labels"].to(self.device)
        
        neg_ids = batch["rejected_input_ids"].to(self.device)
        neg_mask = batch["rejected_attention_mask"].to(self.device)
        neg_labels = batch["rejected_labels"].to(self.device)
        
        # Forward Pass Policy
        policy_pos_logits = self.policy_model(pos_ids, attention_mask=pos_mask).logits
        policy_neg_logits = self.policy_model(neg_ids, attention_mask=neg_mask).logits
        
        policy_pos_logps = self._get_batch_logps(policy_pos_logits, pos_labels)
        policy_neg_logps = self._get_batch_logps(policy_neg_logits, neg_labels)
        
        # Forward Pass Reference
        with torch.no_grad():
            ref_pos_logits = self.ref_model(pos_ids, attention_mask=pos_mask).logits
            ref_neg_logits = self.ref_model(neg_ids, attention_mask=neg_mask).logits
            
            ref_pos_logps = self._get_batch_logps(ref_pos_logits, pos_labels)
            ref_neg_logps = self._get_batch_logps(ref_neg_logits, neg_labels)

        # Compute DPO Loss
        logits = (policy_pos_logps - ref_pos_logps) - (policy_neg_logps - ref_neg_logps)
        losses = -F.logsigmoid(self.args.beta * logits)
        loss = losses.mean()
        
        
        with torch.no_grad():
            chosen_rewards = self.args.beta * (policy_pos_logps - ref_pos_logps)
            rejected_rewards = self.args.beta * (policy_neg_logps - ref_neg_logps)
            reward_acc = (chosen_rewards > rejected_rewards).float().mean()
            
        return loss, {
            "loss/dpo": loss.item(),
            "reward/chosen": chosen_rewards.mean().item(),
            "reward/rejected": rejected_rewards.mean().item(),
            "reward/accuracy": reward_acc.item(),
            "reward/margin": (chosen_rewards - rejected_rewards).mean().item()
        }