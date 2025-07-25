import os
import re
import torch

def load_checkpoint(model, optimizer, path):
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint.get("model_state", {}))
        if optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        epoch = checkpoint.get("epoch", 0) + 1
        print(f"🔁 Loaded checkpoint from {path} (resuming from epoch {epoch})")
        return epoch
    except Exception as e:
        print(f" Failed to load checkpoint: {e}")
        return 0

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)

def find_latest_checkpoint(dir_path="checkpoints"):
    pattern = re.compile(r"bofh_epoch_(\d+)\.pth")
    checkpoints = []

    for f in os.listdir(dir_path):
        full_path = os.path.join(dir_path, f)
        if not os.path.isfile(full_path):
            continue
        match = pattern.match(f)
        if match:
            checkpoints.append((int(match.group(1)), full_path))

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest = checkpoints[0] if checkpoints else (None, None)
    return latest  # → (epoch, path)