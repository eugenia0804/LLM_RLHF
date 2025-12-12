import torch

def collate_fn(batch):
    chosen_ids = torch.tensor([x["chosen_input_ids"] for x in batch], dtype=torch.long)
    chosen_mask = torch.tensor([x["chosen_attention_mask"] for x in batch], dtype=torch.long)
    rejected_ids = torch.tensor([x["rejected_input_ids"] for x in batch], dtype=torch.long)
    rejected_mask = torch.tensor([x["rejected_attention_mask"] for x in batch], dtype=torch.long)
    return chosen_ids, chosen_mask, rejected_ids, rejected_mask