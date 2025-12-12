import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_from_disk

from model_policy import PolicyModel
from model_reward import RewardModel

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data_rlhf")
print(f"Loading data from: {DATA_DIR}")

DEVICE = "cuda:1"
BASE_MODEL_NAME = "gpt2" 
MAX_NEW_TOKENS = 64

# --- Select the model you want to update ---
MODEL_TO_UPDATE = "PPO"  

# --- Define all models ---
MODELS_TO_EVAL = {
    "Base": {"path": None},
    "PPO":  {"path": PROJECT_ROOT + "/results_ppo/final_policy.pt"},
    "DPO":  {"path": PROJECT_ROOT + "/results_dpo/final_dpo_policy.pt"},
    "GRPO": {"path": PROJECT_ROOT + "/results_grpo/best_grpo.policy.pt"},
}


def load_test_prompts(data_dir, num_samples=100):
    """
    Loads the processed dataset from disk and extracts raw prompt text.
    """
    print(f"Loading test data from {data_dir}...")
    try:
        ds = load_from_disk(data_dir)
        test_data = ds["test"]
        
        # If the dataset is huge, shuffle and slice
        if len(test_data) > num_samples:
            test_data = test_data.shuffle(seed=42).select(range(num_samples))
            
        return test_data["prompt_text"]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Did you run the preprocessing script with --data-out-dir data_rlhf?")
        return []


def load_model_custom(checkpoint_path, base_name):
    """
    Loads a policy model. If checkpoint_path is None, returns base model.
    """
    model = PolicyModel(base_name).to(DEVICE)
    if checkpoint_path:
        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def compute_metrics(policy, ref_model, reward_model, tokenizer, prompt_text):
    """
    Generate a response, compute reward, and KL divergence between policy and reference.
    """
    # Ensure input fits GPT-2 context window
    max_model_len = policy.lm.config.n_positions
    safe_input_len = max_model_len - MAX_NEW_TOKENS
    
    inputs = tokenizer(
        prompt_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=safe_input_len
    ).to(DEVICE)
    
    with torch.no_grad():
        # 1. Generate response
        gen_output = policy.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Compute reward
        reward_score = reward_model(gen_output, attention_mask=(gen_output != tokenizer.pad_token_id)).mean().item()
        
        # Compute KL divergence (policy vs reference)
        policy_logits = policy(gen_output).logits
        ref_logits = ref_model(gen_output).logits
        
        kl_div = (policy_logits.log_softmax(dim=-1) - ref_logits.log_softmax(dim=-1)).mean().item()
        
        # Decode only the generated response part
        prompt_len = inputs.input_ids.shape[1]
        response_tokens = gen_output[0][prompt_len:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
    return response_text, reward_score, kl_div


def main():
    # Load test prompts
    test_prompts = load_test_prompts(DATA_DIR, num_samples=100)
    if not test_prompts: 
        print("No test prompts found. Exiting.")
        return

    # Load reference model and reward model
    print("Loading Reference and Reward Models...")
    ref_model = PolicyModel(BASE_MODEL_NAME).to(DEVICE).eval()
    
    reward_model = RewardModel(BASE_MODEL_NAME).to(DEVICE)
    rm_state = torch.load(PROJECT_ROOT + "/results_reward/final_reward_model.pt", map_location=DEVICE)
    
    # Fix keys if needed (from training script logic)
    new_rm_state = {}
    for k, v in rm_state.items():
        new_key = k.replace("lm.transformer", "backbone").replace("reward_head", "head").replace("lm.", "")
        new_rm_state[new_key] = v
    reward_model.load_state_dict(new_rm_state, strict=False)
    reward_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    results = []

    # Load existing CSV and remove old entries for the model we're updating ---
    csv_path = "raw_res/evaluation_metrics_complete.csv"
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing[df_existing["model"] != MODEL_TO_UPDATE]
    else:
        df_existing = pd.DataFrame()

    # Evaluate only the selected model
    config = MODELS_TO_EVAL[MODEL_TO_UPDATE]
    print(f"--- Evaluating {MODEL_TO_UPDATE} ---")
    policy = load_model_custom(config["path"], BASE_MODEL_NAME)

    for prompt in tqdm(test_prompts):
        clean_prompt = prompt.strip()
        resp, reward, kl = compute_metrics(policy, ref_model, reward_model, tokenizer, clean_prompt)
        results.append({
            "model": MODEL_TO_UPDATE,
            "prompt": clean_prompt,
            "response": resp,
            "reward_score": reward,
            "kl_divergence": kl
        })

    del policy
    torch.cuda.empty_cache()

    # Combine old results with new results and save
    df_new = pd.DataFrame(results)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_csv(csv_path, index=False)
    print(f"Updated results for {MODEL_TO_UPDATE} saved to {csv_path}")


if __name__ == "__main__":
    main()
