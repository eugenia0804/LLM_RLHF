import argparse
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import get_scheduler
from tqdm.auto import tqdm

from model_reward import RewardModel, pairwise_ranking_loss  
from vis import plot_train_loss, plot_val_accuracy, plot_grad_norms  
from utils import collate_fn  # Collate function for batching

device = torch.device("cuda:0")  # Set device

@torch.no_grad()
def evaluate_accuracy(model, dataloader):
    # Evaluate model accuracy
    model.eval()
    correct = 0
    total = 0

    limit = len(dataloader) // 2

    with torch.inference_mode():
        for i, (chosen_ids, chosen_mask, rejected_ids, rejected_mask) in enumerate(dataloader):
            if i > limit: break
            
            chosen_ids, chosen_mask = chosen_ids.to(device), chosen_mask.to(device)
            rejected_ids, rejected_mask = rejected_ids.to(device), rejected_mask.to(device)

            with torch.amp.autocast("cuda"):
                input_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
                att_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
                rewards = model(input_ids, att_mask)
                r_chosen, r_rejected = rewards.chunk(2)

            preds = (r_chosen > r_rejected)
            correct += preds.sum().item()
            total += r_chosen.size(0)
        
    return correct / total

def train(args):
    # Load dataset
    ds = load_from_disk(args.data_dir)
    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(ds["validation"], batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=1)

    # Load model
    model = RewardModel(args.model_checkpoint).to(device)
    if hasattr(model, "head"):
        model.head.weight.data.normal_(mean=0.0, std=0.01)
        if model.head.bias is not None:
            model.head.bias.data.zero_()

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    num_update_steps_per_epoch = len(train_loader) // args.accum_steps
    max_train_steps = args.epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),
        num_training_steps=max_train_steps
    )

    # Metrics tracking
    loss_steps, loss_values = [], []
    acc_steps, acc_values = [], []
    grad_norm_values = []
    raw_losses = []
    global_step = 0
    best_val_acc = 0.0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for step, (chosen_ids, chosen_mask, rejected_ids, rejected_mask) in enumerate(progress_bar):
            chosen_ids, chosen_mask = chosen_ids.to(device), chosen_mask.to(device)
            rejected_ids, rejected_mask = rejected_ids.to(device), rejected_mask.to(device)

            with torch.amp.autocast("cuda"):
                input_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
                att_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
                rewards = model(input_ids, att_mask)
                r_chosen, r_rejected = rewards.chunk(2)
                loss = pairwise_ranking_loss(r_chosen, r_rejected) / args.accum_steps

            scaler.scale(loss).backward()

            raw_losses.append(loss.item() * args.accum_steps)
            progress_bar.set_postfix({"loss": raw_losses[-1]})

            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % 5 == 0:
                    window = raw_losses[-20:] if len(raw_losses) > 100 else raw_losses
                    moving_avg = sum(window) / len(window)
                    loss_steps.append(global_step)
                    loss_values.append(moving_avg)
                    grad_norm_values.append(grad_norm.item())
                    plot_train_loss(loss_steps, loss_values, args.output_dir)
                    plot_grad_norms(loss_steps, grad_norm_values, args.output_dir)

                # Validation
                if global_step % args.val_steps == 0:
                    val_acc = evaluate_accuracy(model, val_loader)
                    acc_steps.append(global_step)
                    acc_values.append(val_acc)
                    plot_val_accuracy(acc_steps, acc_values, args.output_dir)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        save_path = os.path.join(args.output_dir, "best_reward_model.pt")
                        torch.save(model.state_dict(), save_path)

    # Save final model
    final_path = os.path.join(args.output_dir, "final_reward_model.pt")
    torch.save(model.state_dict(), final_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_reward")
    parser.add_argument("--model-checkpoint", type=str, default="gpt2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="results_reward_gpt2")
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--val-steps", type=int, default=20)
    
    args = parser.parse_args()
    train(args)
