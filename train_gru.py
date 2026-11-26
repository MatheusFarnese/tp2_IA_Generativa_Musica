import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prepare_dataset import WindowDataset, make_song_balanced_sampler


# ============================================================
# Model
# ============================================================

class GRULanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size=502,
        pad_id=502,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 1,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)

        outputs, _ = self.gru(embedded)

        outputs = self.dropout(outputs)

        return self.output(outputs)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# Data loaders
# ============================================================

def make_loader(
    dataset,
    batch_size,
    training,
    num_workers,
    device,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }

    if training:
        kwargs["sampler"] = make_song_balanced_sampler(dataset)
    else:
        kwargs["shuffle"] = False

    return DataLoader(**kwargs)


# ============================================================
# Training / evaluation epoch
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    pad_id,
    grad_clip,
):
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in loader:

        x = batch["input_ids"].to(
            device,
            non_blocking=True,
        )

        y = batch["target_ids"].to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            logits = model(x)

            loss_sum = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            if training:
                loss_sum.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

                optimizer.step()

        valid_mask = y.ne(pad_id)
        valid_count = int(valid_mask.sum().item())

        if valid_count > 0:

            predictions = logits.argmax(dim=-1)

            total_correct += int(
                ((predictions == y) & valid_mask).sum().item()
            )

            total_tokens += valid_count
            total_loss += loss_sum.detach().item()

    if total_tokens == 0:
        raise RuntimeError("No valid target tokens in this split.")

    return {
        "loss": total_loss / total_tokens,
        "accuracy": total_correct / total_tokens,
        "tokens": total_tokens,
    }


# ============================================================
# Utilities
# ============================================================

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Train one context length
# ============================================================

def train_context(args, context, device):

    set_seed(args.seed)

    data_dir = args.data_dir / f"context_{context}"
    out_dir = args.output_dir / f"context_{context}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    datasets = {}

    for split in ("train", "val", "test"):

        path = data_dir / f"{split}.pt"

        if not path.is_file():
            raise FileNotFoundError(path)

        datasets[split] = WindowDataset(path)

        if len(datasets[split]) == 0:
            raise ValueError(
                f"{split} dataset is empty for context {context}"
            )

    pad_id = int(datasets["train"].payload["pad_id"])

    for split, dataset in datasets.items():

        if int(dataset.payload["context_length"]) != context:
            raise ValueError(
                f"Incorrect context length in {split}"
            )

        if int(dataset.payload["pad_id"]) != pad_id:
            raise ValueError(
                f"Inconsistent PAD ID in {split}"
            )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    loaders = {

        "train": make_loader(
            datasets["train"],
            args.batch_size,
            True,
            args.num_workers,
            device,
        ),

        "val": make_loader(
            datasets["val"],
            args.batch_size,
            False,
            args.num_workers,
            device,
        ),

        "test": make_loader(
            datasets["test"],
            args.batch_size,
            False,
            args.num_workers,
            device,
        ),
    }

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GRULanguageModel(
        vocab_size=args.vocab_size,
        pad_id=pad_id,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        reduction="sum",
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "context_length": context,
        "vocab_size": args.vocab_size,
        "pad_id": pad_id,
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.grad_clip,
        "seed": args.seed,
        "sampler": "equal expected sampling weight per song",
        "loss": "token-level cross entropy; padding ignored",
    }

    save_json(out_dir / "config.json", config)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    history = []

    checkpoint_path = out_dir / "best_model.pt"
    save_path = out_dir / "last_model.pt"

    print(f"\n{'=' * 60}")
    print(f"Context length: {context}")
    print(f"Device: {device}")

    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset)} windows")

    print(f"{'=' * 60}")

    for epoch in range(1, args.epochs + 1):

        start = time.time()

        train_metrics = run_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            pad_id,
            args.grad_clip,
        )

        val_metrics = run_epoch(
            model,
            loaders["val"],
            None,
            criterion,
            device,
            pad_id,
            args.grad_clip,
        )

        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - start

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": current_lr,
            "seconds": elapsed,
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Train acc: {train_metrics['accuracy']:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # Checkpointing
        # ----------------------------------------------------

        if train_metrics["loss"] < best_loss - args.min_delta:

            best_loss = train_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_loss": best_loss,
                },
                checkpoint_path,
            )

        else:
            stale_epochs += 1

        with open(
            out_dir / "history.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(history[0].keys()),
            )

            writer.writeheader()
            writer.writerows(history)

        save_json(
            out_dir / "history.json",
            {"epochs": history},
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if stale_epochs >= args.patience:

            print(
                f"Early stopping after {args.patience} "
                "epochs without improvement."
            )

            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
        },
        save_path,
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    if not checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was produced.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    test_metrics = run_epoch(
        model,
        loaders["test"],
        None,
        criterion,
        device,
        pad_id,
        args.grad_clip,
    )

    result = {
        "context_length": context,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_valid_tokens": test_metrics["tokens"],
        "epochs_trained": len(history),
        "best_checkpoint": str(checkpoint_path),
    }

    save_json(
        out_dir / "result.json",
        result,
    )

    print("\nFinal results:")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation loss: {best_loss:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("prepared"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/gru"),
    )

    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[128, 256, 512],
    )

    parser.add_argument("--vocab-size", type=int, default=502)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if args.num_layers < 1 or args.batch_size < 1:
        raise ValueError("num-layers and batch-size must be positive.")

    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for context in args.contexts:

        result = train_context(
            args,
            context,
            device,
        )

        results.append(result)

    save_json(
        args.output_dir / "comparison.json",
        {"experiments": results},
    )

    print("\nExperiment comparison:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()