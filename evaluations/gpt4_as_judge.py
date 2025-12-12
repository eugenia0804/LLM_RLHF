import pandas as pd
import os
import json
import random
from tqdm import tqdm
from openai import OpenAI

# --- CONFIGURATION ---
INPUT_FILE = "raw_res/evaluation_metrics_complete.csv"
OUTPUT_FILE = "raw_res/win_rate_results.csv"
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

JUDGE_PROMPT = """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt.
Choose the assistant that answers better. 

[User Prompt]
{prompt}

[Assistant A]
{answer_a}

[Assistant B]
{answer_b}

Output JSON only:
{{
  "winner": "A" or "B" or "Tie",
  "reasoning": "short explanation"
}}
"""

def evaluate_pair(prompt, text_a, text_b, model_a, model_b):
    # Randomly swap to avoid position bias
    if random.choice([True, False]):
        t1, t2 = text_a, text_b
        m1, m2 = model_a, model_b
    else:
        t1, t2 = text_b, text_a
        m1, m2 = model_b, model_a

    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(prompt=prompt, answer_a=t1, answer_b=t2)}],
            response_format={"type": "json_object"}
        )
        result = json.loads(completion.choices[0].message.content)
        winner_tag = result.get("winner", "Tie")
        
        real_winner = m1 if winner_tag == "A" else (m2 if winner_tag == "B" else "Tie")
        return real_winner
    except Exception as e:
        print(f"Error: {e}")
        return "Error"

def main():
    df = pd.read_csv(INPUT_FILE)
    
    # Isolate Base Model dataframe
    base_df = df[df['model'] == 'Base'][['prompt', 'response']].rename(columns={'response': 'response_base'})
    
    results = []
    
    # Compare each fine-tuned model against Base
    for model_name in ["PPO", "DPO", "GRPO"]:
        print(f"--- Judging {model_name} vs Base ---")
        model_df = df[df['model'] == model_name][['prompt', 'response']]
        
        # Merge on prompt to get pairs
        merged = pd.merge(base_df, model_df, on='prompt', suffixes=('_base', '_ft'))
        
        for _, row in tqdm(merged.iterrows(), total=len(merged)):
            winner = evaluate_pair(
                row['prompt'], 
                row['response_base'], 
                row['response'], 
                "Base", 
                model_name
            )
            results.append({"model": model_name, "winner": winner})

    # Save and Print Summary
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n=== Win Rates vs Base Model ===")
    for model in ["PPO", "DPO", "GRPO"]:
        wins = len(res_df[(res_df['model'] == model) & (res_df['winner'] == model)])
        total = len(res_df[res_df['model'] == model])
        if total > 0:
            print(f"{model}: {wins/total:.1%} ({wins}/{total})")

if __name__ == "__main__":
    main()