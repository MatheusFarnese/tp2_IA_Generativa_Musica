"""Tokenize split MIDI files as a single synchronized REMI+ stream.

MidiTok 3.x: REMI + Program + TimeSignature tokens. With
one_token_stream_for_programs=True, all instruments are serialized into one
chronologically ordered stream, with Program tokens identifying instruments.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from miditok import REMI, TokenizerConfig

MIDI_EXTENSIONS = {".mid", ".midi"}


def collect_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MIDI_EXTENSIONS)


def extract_ids(encoded) -> list[int]:
    """Extract IDs from the single TokSequence expected in REMI+ single-stream mode."""
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    # Some MidiTok versions wrap one sequence in a list.
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and hasattr(encoded[0], "ids"):
        return list(encoded[0].ids)
    raise TypeError(
        "Expected one TokSequence from REMI+ single-stream mode; "
        f"received {type(encoded).__name__}. Check MidiTok version/configuration."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="Contains train/, val/, and test/ folders")
    parser.add_argument("--output", type=Path, default=Path("processed"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # REMI+ is obtained by enabling programs, one synchronized token stream,
    # and time signatures in the standard REMI tokenizer.
    config = TokenizerConfig(
        num_velocities=16,
        use_programs=True,
        one_token_stream_for_programs=True,
        program_changes=False,
        use_tempos=True,
        use_time_signatures=True,
        use_pitch_bends=False,
        use_sustain_pedals=False,
        beat_res={(0, 4): 8, (4, 12): 4},
    )
    tokenizer = REMI(config)
    tokenizer.save_pretrained(args.output / "remi_plus_tokenizer")
    #tokenizer.save(args.output / "remi_plus_tokenizer")

    rows: list[dict] = []
    for split in ("train", "val", "test"):
        folder = args.dataset_dir / split
        if not folder.is_dir():
            raise SystemExit(f"Missing split folder: {folder}")
        out_folder = args.output / split
        out_folder.mkdir(parents=True, exist_ok=True)
        for midi_path in collect_files(folder):
            try:
                # Passing the path lets MidiTok load it through its supported backend.
                encoded = tokenizer(midi_path)
                ids = extract_ids(encoded)
                #token_strings = tokenizer.id_to_piece(ids)
                id_to_token = {v: k for k, v in tokenizer.vocab.items()}
                token_strings = [id_to_token[token_id] for token_id in ids]
                payload = {
                    "source_file": midi_path.name,
                    "split": split,
                    "ids": ids,
                    "tokens": token_strings,
                }
                (out_folder / f"{midi_path.stem}.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                rows.append({"file": midi_path.name, "split": split, "n_tokens": len(ids), "error": ""})
            except Exception as exc:
                rows.append({
                    "file": midi_path.name, "split": split, "n_tokens": "",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    csv_path = args.output / "sequence_lengths.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "split", "n_tokens", "error"])
        writer.writeheader()
        writer.writerows(rows)

    successful = [r for r in rows if r["n_tokens"] != ""]
    lengths = [int(r["n_tokens"]) for r in successful]
    summary = {
        "tokenization": "REMI+",
        "miditok_version_note": "Pin miditok==3.0.1 for reproducibility",
        "single_synchronized_stream": True,
        "use_programs": True,
        "one_token_stream_for_programs": True,
        "use_time_signatures": True,
        "vocabulary_size": len(tokenizer),
        "files_tokenized": len(successful),
        "files_failed": len(rows) - len(successful),
        "total_tokens": sum(lengths),
        "mean_tokens_per_file": sum(lengths) / len(lengths) if lengths else None,
        "median_tokens_per_file": sorted(lengths)[len(lengths)//2] if lengths else None,
        "min_tokens": min(lengths) if lengths else None,
        "max_tokens": max(lengths) if lengths else None,
    }
    (args.output / "tokenization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Per-file results: {csv_path}")


if __name__ == "__main__":
    main()
