import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

# Policy Model for PPO: wraps a causal LM (Actor) and a Value Head (Critic)
class PolicyModel(nn.Module):
    def __init__(self, model_checkpoint="gpt2"):
        super().__init__()
        
        # Load model configuration
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        
        # Actor: Causal LM
        self.lm = AutoModelForCausalLM.from_pretrained(model_checkpoint, config=self.config)
        
        # Critic: Value Head projecting hidden states to scalar values
        hidden_size = self.config.n_embd
        self.v_head = nn.Linear(hidden_size, 1)
        
        # Initialize value head
        self.v_head.weight.data.normal_(mean=0.0, std=0.2)
        self.v_head.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Enable hidden states for value computation
        kwargs["output_hidden_states"] = True
        
        # Forward pass through actor
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits
        
        # Compute value estimates from last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        values = self.v_head(last_hidden_state).squeeze(-1)
        
        # Return both policy logits and value estimates
        return type('PolicyOutput', (object,), {
            'logits': logits,
            'values': values
        })()

    def generate(self, *args, **kwargs):
        # Proxy to Hugging Face LM generate method for convenience
        return self.lm.generate(*args, **kwargs)
