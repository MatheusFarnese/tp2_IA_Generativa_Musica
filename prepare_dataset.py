#!/usr/bin/env python3
"""Prepare fixed-length next-token windows from MidiTok JSON files.

Input: processed/{train,val,test}/*.json, each containing an integer list in `ids`.
Output: prepared/context_{128,256,512}/{train,val,test}.pt and summary.json.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

SPLITS = ("train", "val", "test")
DEFAULT_CONTEXTS = (128, 256, 512)


def load_sequences(processed_dir: Path, vocab_size: int):
    sequences: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    report = {s: {"files_found": 0, "files_loaded": 0, "files_skipped": []} for s in SPLITS}
    for split in SPLITS:
        split_dir = processed_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for path in sorted(split_dir.glob("*.json")):
            report[split]["files_found"] += 1
            try:
                with path.open(encoding="utf-8") as f:
                    obj = json.load(f)
                ids = obj.get("ids")
                if not isinstance(ids, list):
                    raise ValueError("Missing or invalid 'ids' list")
                if not all(isinstance(x, int) and not isinstance(x, bool) for x in ids):
                    raise ValueError("'ids' must contain only integers")
                bad = next((x for x in ids if x < 0 or x >= vocab_size), None)
                if bad is not None:
                    raise ValueError(f"Token ID {bad} outside [0, {vocab_size - 1}]")
                declared = obj.get("split")
                if declared is not None and declared != split:
                    raise ValueError(f"JSON declares split={declared!r}, but is stored in {split_dir}")
                if len(ids) < 2:
                    report[split]["files_skipped"].append({"file": path.name, "reason": "Fewer than 2 tokens"})
                    continue
                sequences[split].append({"source_file": str(obj.get("source_file", path.name)), "ids": ids})
                report[split]["files_loaded"] += 1
            except Exception as exc:
                report[split]["files_skipped"].append({"file": path.name, "reason": str(exc)})
    return sequences, report


def get_window_starts(num_transitions: int, context: int, stride: int, overlapping: bool):
    if num_transitions <= 0:
        return []
    if not overlapping:
        return list(range(0, num_transitions, stride))
    if num_transitions <= context:
        return [0]
    last_start = num_transitions - context
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def make_payload(songs, split: str, context: int, pad_id: int, train_stride: int):
    is_train = split == "train"
    stride = train_stride if is_train else context
    inputs, targets, lengths, song_indices, window_files = [], [], [], [], []
    source_files = [song["source_file"] for song in songs]
    for song_idx, song in enumerate(songs):
        ids = song["ids"]
        for start in get_window_starts(len(ids) - 1, context, stride, is_train):
            x = ids[start:start + context]
            y = ids[start + 1:start + context + 1]
            valid = len(y)
            inputs.append(x + [pad_id] * (context - len(x)))
            targets.append(y + [pad_id] * (context - len(y)))
            lengths.append(valid)
            song_indices.append(song_idx)
            window_files.append(song["source_file"])
    n = len(inputs)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long) if n else torch.empty((0, context), dtype=torch.long),
        "target_ids": torch.tensor(targets, dtype=torch.long) if n else torch.empty((0, context), dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "song_index": torch.tensor(song_indices, dtype=torch.long),
        "source_files": source_files,
        "window_song_files": window_files,
        "context_length": context,
        "pad_id": pad_id,
    }


def summarize(payload):
    counts = Counter(payload["song_index"].tolist())
    valid = int(payload["lengths"].sum().item())
    return {
        "songs": len(payload["source_files"]),
        "windows": int(payload["input_ids"].shape[0]),
        "total_valid_targets": valid,
        "padding_positions": int(payload["target_ids"].numel() - valid),
        "min_windows_per_song": min(counts.values()) if counts else 0,
        "max_windows_per_song": max(counts.values()) if counts else 0,
        "songs_without_windows": len(payload["source_files"]) - len(counts),
    }


class WindowDataset(Dataset):
    """Dataset wrapper; each item is one fixed-length window."""
    def __init__(self, path: str | Path):
        self.payload = torch.load(path, map_location="cpu", weights_only=False)
    def __len__(self):
        return self.payload["input_ids"].shape[0]
    def __getitem__(self, index):
        return {
            "input_ids": self.payload["input_ids"][index],
            "target_ids": self.payload["target_ids"][index],
            "length": self.payload["lengths"][index],
            "song_index": self.payload["song_index"][index],
        }


def make_song_balanced_sampler(dataset: WindowDataset):
    """Equal total sampling weight per song; samples len(dataset) windows per epoch."""
    indices = dataset.payload["song_index"]
    if len(indices) == 0:
        raise ValueError("Cannot sample an empty dataset")
    counts = torch.bincount(indices)
    weights = torch.tensor([1.0 / counts[int(i)] for i in indices], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def main():
    p = argparse.ArgumentParser(description="Create next-token windows from MidiTok JSON sequences")
    p.add_argument("--processed-dir", type=Path, default=Path("processed"))
    p.add_argument("--output-dir", type=Path, default=Path("prepared"))
    p.add_argument("--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS))
    p.add_argument("--vocab-size", type=int, default=502, help="Real vocabulary size; IDs are 0..501")
    p.add_argument("--train-stride", type=int, default=None, help="Default: half the context length")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.vocab_size <= 0 or any(c <= 0 for c in args.contexts):
        p.error("Vocabulary size and context lengths must be positive")
    pad_id = args.vocab_size
    sequences, loading_report = load_sequences(args.processed_dir, args.vocab_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"processed_dir": str(args.processed_dir), "output_dir": str(args.output_dir),
               "vocab_size": args.vocab_size, "pad_id": pad_id,
               "loading_report": loading_report, "contexts": {}}
    for context in args.contexts:
        stride = args.train_stride if args.train_stride is not None else max(1, context // 2)
        if not 1 <= stride <= context:
            p.error("--train-stride must be between 1 and the context length")
        out_dir = args.output_dir / f"context_{context}"
        out_dir.mkdir(parents=True, exist_ok=True)
        context_info = {"train_stride": stride, "validation_test_stride": context}
        for split in SPLITS:
            payload = make_payload(sequences[split], split, context, pad_id, stride)
            out_path = out_dir / f"{split}.pt"
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(f"{out_path} exists; use --overwrite to replace it")
            torch.save(payload, out_path)
            context_info[split] = summarize(payload)
            context_info[split]["output_file"] = str(out_path)
        summary["contexts"][str(context)] = context_info
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; use --overwrite to replace it")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDone. Summary: {summary_path}")

if __name__ == "__main__":
    main()
