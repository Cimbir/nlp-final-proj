import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TextEncoder
from losses import InfoNCELoss
from data import TripletDataset
from tokenizer import load_tokenizer

MODEL_NAME = "TextEncoder-4L-256d-nlp-tuned-neg"

BATCH_SIZE      = 256
LR              = 1e-3
WEIGHT_DECAY    = 0.01
WARMUP_STEPS    = 500
EPOCHS          = 20
TEMPERATURE     = 0.07
QUERY_MAX_LEN   = 64
PASSAGE_MAX_LEN = 256

TOKENIZER_NAME = "data/processed/bpe_tokenizer.json"
# Need Combine book dataset(which is smaller) with other datasets
TRAIN_PATHS = [
    "data/processed/msmarco_train_triplets.jsonl",
    "data/processed/book_train_triplets.jsonl",
    "data/processed/squad_train_triplets.jsonl",
    # "data/processed/nq_train_triplets.jsonl",
]
VAL_PATHS = [
    # "data/processed/msmarco_val_triplets.jsonl",
    "data/processed/book_val_triplets.jsonl",
]
CHECKPOINT_DIR = "checkpoints"
EARLY_STOP_PAT = 4
LOG_EVERY = 100


def validate(model, loader, criterion, device) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            q_emb = model(batch["query_ids"].to(device), batch["query_mask"].to(device))
            p_emb = model(batch["pos_ids"].to(device),   batch["pos_mask"].to(device))
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                loss = criterion(q_emb, p_emb)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def train() -> dict:
    """
    Train the TextEncoder and return a history dict for plotting:
        {
            "train_loss": [float, ...],   # one value per step
            "train_lr":   [float, ...],   # one value per step
            "train_step": [int,   ...],   # global step index
            "val_loss":   [float, ...],   # one value per epoch
            "val_epoch":  [int,   ...],   # epoch index (1-based)
        }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model_name = MODEL_NAME
    ckpt_dir = Path(CHECKPOINT_DIR) / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(TOKENIZER_NAME)

    print("Loading datasets")
    train_datasets = [
        TripletDataset(p, tokenizer, QUERY_MAX_LEN, PASSAGE_MAX_LEN)
        for p in TRAIN_PATHS
    ]
    train_ds = ConcatDataset(train_datasets)
    val_datasets = [
        TripletDataset(p, tokenizer, QUERY_MAX_LEN, PASSAGE_MAX_LEN) for p in VAL_PATHS
    ]
    val_ds = ConcatDataset(val_datasets)
    print(f"  train: {len(train_ds):,}  |  val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = TextEncoder(vocab_size=len(tokenizer)).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = InfoNCELoss(temperature=TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    log_path = ckpt_dir / "train.log"
    log_file = open(log_path, "w", encoding="utf-8")

    history = {
        "train_loss": [],
        "train_lr": [],
        "train_step": [],
        "val_loss": [],
        "val_epoch": [],
    }
    best_val_loss = float("inf")
    patience = 0
    global_step = 0

    try:
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for step, batch in enumerate(train_loader):
                q_ids  = batch["query_ids"].to(device)
                q_mask = batch["query_mask"].to(device)
                p_ids  = batch["pos_ids"].to(device)
                p_mask = batch["pos_mask"].to(device)
                n_ids  = batch["neg_ids"].to(device)
                n_mask = batch["neg_mask"].to(device)

                with torch.amp.autocast(
                    device_type=device.type, enabled=(device.type == "cuda")
                ):
                    q_emb = model(q_ids, q_mask)
                    p_emb = model(p_ids, p_mask)
                    n_emb = model(n_ids, n_mask)
                    loss  = criterion(q_emb, p_emb, n_emb)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running_loss += loss.item()
                history["train_loss"].append(loss.item())
                history["train_lr"].append(scheduler.get_last_lr()[0])
                history["train_step"].append(global_step)

                if (step + 1) % LOG_EVERY == 0:
                    avg = running_loss / LOG_EVERY
                    msg = (
                        f"Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(train_loader)} "
                        f"| Loss {avg:.4f} | LR {scheduler.get_last_lr()[0]:.2e}"
                    )
                    print(msg)
                    log_file.write(msg + "\n")
                    log_file.flush()
                    running_loss = 0.0

            val_loss = validate(model, val_loader, criterion, device)
            epoch_msg = f"Epoch {epoch+1} | Val loss: {val_loss:.4f}"
            print(epoch_msg)
            log_file.write(epoch_msg + "\n")
            log_file.flush()
            history["val_loss"].append(val_loss)
            history["val_epoch"].append(epoch + 1)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_dir / f"epoch_{epoch+1}.pt",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    ckpt_dir / "best_model.pt",
                )
                print(f"  New best model saved (val_loss={val_loss:.4f})")
            else:
                patience += 1
                if patience >= EARLY_STOP_PAT:
                    print("Early stopping triggered.")
                    log_file.write("Early stopping triggered.\n")
                    break

    finally:
        log_file.close()

    print()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {ckpt_dir}/best_model.pt")

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {ckpt_dir}/training_history.json")

    return history


def load_model(
    checkpoint_path: str = "checkpoints/TextEncoder-4L-256d-scratch/best_model.pt",
    device: str = "cpu",
) -> tuple:
    """Load a trained TextEncoder and its tokenizer. Returns (model, tokenizer)"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    tokenizer = load_tokenizer(TOKENIZER_NAME)
    model = TextEncoder(vocab_size=len(tokenizer))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, tokenizer


if __name__ == "__main__":
    train()
