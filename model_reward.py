import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# Reward Model with a scalar head for RLHF scoring
class RewardModel(nn.Module):
    def __init__(self, checkpoint_name):
        super().__init__()
        
        # Load base model configuration
        self.config = AutoConfig.from_pretrained(checkpoint_name)
        
        # Determine if model is decoder (GPT-style) or encoder (BERT-style)
        self.is_decoder = False
        if any(k in self.config.architectures[0].lower() for k in ["gpt", "opt", "llama", "mistral"]):
            self.is_decoder = True
            
        # Load backbone transformer: extract hidden states for scoring
        self.backbone = AutoModel.from_pretrained(checkpoint_name)
        
        # Define a linear head to produce a single reward scalar per sequence
        self.head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        # Forward through transformer backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Pool hidden states into a single sentence embedding
        if self.is_decoder:
            # take last non-padding token
            if attention_mask is not None:
                last_token_indices = attention_mask.sum(dim=1) - 1
                last_token_indices = last_token_indices.clamp(min=0)
                sentence_embedding = last_hidden_state[
                    torch.arange(last_hidden_state.size(0), device=last_hidden_state.device),
                    last_token_indices
                ]
            else:
                # Default to last token if no mask
                sentence_embedding = last_hidden_state[:, -1, :]
        else:
            # Encoder-style: use [CLS] token embedding
            sentence_embedding = last_hidden_state[:, 0, :]

        # Project pooled embedding to a scalar reward
        rewards = self.head(sentence_embedding)
        
        # Squeeze to shape [batch]
        return rewards.squeeze(-1)

# Pairwise ranking loss for reward model training
def pairwise_ranking_loss(chosen_rewards, rejected_rewards):
    diff = chosen_rewards - rejected_rewards
    return -torch.nn.functional.logsigmoid(diff).mean()
