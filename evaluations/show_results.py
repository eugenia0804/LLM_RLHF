import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_FILE = "raw_res/evaluation_metrics_complete.csv"

def main():
    df = pd.read_csv(INPUT_FILE)

    # 1. Generate Summary Table (Pareto Frontier Data)
    summary = df.groupby("model").agg({
        "reward_score": ["mean", "std"],
        "kl_divergence": ["mean", "std"]
    }).reset_index()
    
    # Flatten columns for clean printing
    summary.columns = ['Model', 'Reward_Mean', 'Reward_Std', 'KL_Mean', 'KL_Std']
    
    print("\n=== Quantitative Evaluation Table (Pareto Frontier) ===")
    print(summary.to_string(index=False))
    
    # Save table to CSV for your report
    summary.to_csv("raw_res/pareto_table.csv", index=False)

    # 2. Plot: Reward Distributions (KDE)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="reward_score", hue="model", fill=True, common_norm=False, palette="viridis", alpha=0.4)
    plt.title("Reward Model Score Distributions")
    plt.xlabel("Reward Score")
    plt.grid(True, alpha=0.3)
    plt.savefig("raw_res/reward_distribution.png", dpi=300)

    # 3. Plot: Pareto Frontier (Scatter)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=summary, x="KL_Mean", y="Reward_Mean", hue="Model", s=200, palette="viridis", style="Model")
    
    # Add labels
    for i, row in summary.iterrows():
        plt.text(row['KL_Mean'] + 0.002, row['Reward_Mean'] + 0.01, row['Model'], fontsize=12, fontweight='bold')
        
    plt.title("Pareto Frontier: Reward Maximization vs. KL Constraint")
    plt.xlabel("KL Divergence (Drift from Base)")
    plt.ylabel("Average Reward Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("raw_res/pareto_frontier.png", dpi=300)

if __name__ == "__main__":
    main()