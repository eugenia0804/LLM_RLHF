import matplotlib.pyplot as plt
import os

def plot_train_loss(steps, losses, output_dir, log_scale=False):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="Training Loss (Moving Avg)")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    if log_scale:
        plt.yscale("log")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train_loss_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] Updated: {out_path}")


def plot_val_accuracy(steps, accuracies, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, accuracies, label="Validation Accuracy")
    plt.xlabel("Global Step")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "val_accuracy_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] Updated: {out_path}")


def plot_grad_norms(steps, norms, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, norms, label="Gradient Norm", color='orange')

    plt.xlabel("Global Step")
    plt.ylabel("Norm")
    plt.title("Gradient Norm Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "grad_norm_curve.png")
    plt.savefig(out_path)
    plt.close()


def plot_rlhf_stats(steps, rewards, kls, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Mean Reward
    plt.subplot(1, 2, 1)
    plt.plot(steps, rewards, label="Mean Reward", color='green')
    plt.xlabel("Step")
    plt.title("Mean Reward Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 2: KL Divergence
    plt.subplot(1, 2, 2)
    plt.plot(steps, kls, label="KL Divergence", color='orange')
    plt.xlabel("Step")
    plt.title("KL Divergence Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "rlhf_metrics.png"))
    plt.close()