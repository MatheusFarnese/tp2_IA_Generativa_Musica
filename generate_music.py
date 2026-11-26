import argparse
import json
import random
from pathlib import Path

import torch
from miditok import REMI

from train_gru import GRULanguageModel


# ============================================================
# Tokenizer and model
# ============================================================

def load_tokenizer(config_path):
    tokenizer = REMI(params=config_path)

    print(f"Tokenizer vocabulary: {len(tokenizer)}")

    return tokenizer


def load_model(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    model = GRULanguageModel(
        vocab_size=config["vocab_size"],
        pad_id=config["pad_id"],
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['best_val_loss']:.4f}")

    return model, config


# ============================================================
# Initial sequence
# ============================================================

def create_initial_sequence(
    tokenizer,
    tempo=110,
    program=0,
    time_signature="4/4",
):
    """
    Build a short structural prefix.

    The prefix contains no melody copied from the training set.
    """

    vocab = tokenizer.vocab

    required_tokens = [
        "Bar_None",
        f"TimeSig_{time_signature}",
        "Position_0",
        f"Program_{program}",
    ]

    for token in required_tokens:
        if token not in vocab:
            raise ValueError(
                f"Token {token!r} not found in tokenizer vocabulary."
            )

    tempo_tokens = {
        token: token_id
        for token, token_id in vocab.items()
        if token.startswith("Tempo_")
    }

    if not tempo_tokens:
        raise ValueError("No Tempo tokens found in vocabulary.")

    def tempo_value(token):
        return float(token.split("_", 1)[1])

    selected_tempo = min(
        tempo_tokens,
        key=lambda token: abs(tempo_value(token) - tempo),
    )

    prefix_tokens = [
        "Bar_None",
        f"TimeSig_{time_signature}",
        "Position_0",
        selected_tempo,
        f"Program_{program}",
    ]

    prefix_ids = [vocab[token] for token in prefix_tokens]

    return prefix_ids


def create_primer_from_midi(tokenizer, midi_path, n_bars=4):
    sequence = tokenizer(Path(midi_path))

    if isinstance(sequence, list):
        if len(sequence) != 1:
            raise ValueError(
                "Expected a single token stream for REMI+."
            )
        sequence = sequence[0]

    ids = list(sequence.ids)

    if not ids:
        raise ValueError("The primer MIDI produced no tokens.")

    bar_positions = [
        i for i, token_id in enumerate(ids)
        if tokenizer[token_id].startswith("Bar_")
    ]

    if len(bar_positions) > n_bars:
        ids = ids[:bar_positions[n_bars]]

    return ids


# ============================================================
# Sampling & Decoding Helpers
# ============================================================

def get_forbidden_ids(vocab):
    """
    Exclude special tokens that were not used as targets
    during the training procedure.
    """
    return [
        token_id
        for token, token_id in vocab.items()
        if token.split("_", 1)[0] in {
            "PAD", "BOS", "EOS", "MASK"
        }
    ]


def sample_next_token(
    logits,
    temperature=1.0,
    top_k=20,
    forbidden_ids=None,
):
    """
    Sample one token using temperature and top-k filtering.
    """

    logits = logits.clone()

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    logits = logits / temperature

    if forbidden_ids:
        logits[forbidden_ids] = float("-inf")

    if top_k > 0:
        k = min(top_k, logits.numel())

        threshold = torch.topk(logits, k).values[-1]

        logits[logits < threshold] = float("-inf")

    probabilities = torch.softmax(logits, dim=-1)

    next_token = torch.multinomial(
        probabilities,
        num_samples=1,
    )

    return next_token.item()


# ============================================================
# Autoregressive generation (Random Sampling & Beam Search)
# ============================================================

@torch.no_grad()
def generate_random(
    model,
    tokenizer,
    initial_ids,
    max_new_tokens,
    context_length,
    temperature,
    top_k,
    device,
):
    generated_ids = list(initial_ids)
    forbidden_ids = get_forbidden_ids(tokenizer.vocab)

    print("\nGenerating music with Random Sampling...")

    for step in range(max_new_tokens):

        context = generated_ids[-context_length:]

        input_tensor = torch.tensor(
            context,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        logits = model(input_tensor)

        next_logits = logits[0, -1, :]

        next_token = sample_next_token(
            logits=next_logits,
            temperature=temperature,
            top_k=top_k,
            forbidden_ids=forbidden_ids,
        )

        generated_ids.append(next_token)

        if (step + 1) % 100 == 0:
            print(
                f"Generated {step + 1}/{max_new_tokens} tokens"
            )

    print(f"\nGeneration finished: {len(generated_ids)} total tokens.")

    return generated_ids


@torch.no_grad()
def generate_beam_search(
    model,
    tokenizer,
    initial_ids,
    max_new_tokens,
    context_length,
    beam_width,
    device,
):
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1.")

    forbidden_ids = get_forbidden_ids(tokenizer.vocab)

    beams = [(list(initial_ids), 0.0)]

    print(f"\nGenerating music with Beam Search (beam_width={beam_width})...")

    for step in range(max_new_tokens):
        contexts = [seq[-context_length:] for seq, _ in beams]

        input_tensor = torch.tensor(
            contexts,
            dtype=torch.long,
            device=device,
        )

        logits = model(input_tensor)
        next_logits = logits[:, -1, :]

        if forbidden_ids:
            next_logits[:, forbidden_ids] = float("-inf")

        log_probs = torch.log_softmax(next_logits, dim=-1)
        k = min(beam_width, log_probs.size(-1))

        all_candidates = []
        for beam_idx, (seq, score) in enumerate(beams):
            top_log_probs, top_indices = torch.topk(log_probs[beam_idx], k=k)
            for token_log_prob, token_id in zip(
                top_log_probs.tolist(), top_indices.tolist()
            ):
                all_candidates.append(
                    (seq + [token_id], score + token_log_prob)
                )

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

        if (step + 1) % 100 == 0:
            print(
                f"Generated {step + 1}/{max_new_tokens} tokens "
                f"(best log-prob: {beams[0][1]:.2f})"
            )

    best_sequence, best_score = beams[0]
    print(
        f"\nGeneration finished: {len(best_sequence)} total tokens "
        f"(best log-prob: {best_score:.2f})."
    )

    return best_sequence


# ============================================================
# MIDI export
# ============================================================

def save_midi(tokenizer, generated_ids, output_path):
    output_path = Path(output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Number of generated tokens: {len(generated_ids)}")

    error_ratio = tokenizer.tokens_errors(generated_ids)

    midi = tokenizer.decode(generated_ids)

    midi.dump_midi(str(output_path))

    print(f"MIDI saved to: {output_path}")

    return midi, error_ratio


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate music using a trained REMI+ GRU."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        default=Path("processed/tokenizer.json"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated/generated_music.mid"),
    )

    parser.add_argument(
        "--decoding",
        choices=["random", "beam"],
        default="random",
        help="Decoding strategy: 'random' (top-k sampling) or 'beam' (beam search).",
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Number of beams to keep at each step when --decoding beam is used.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--tempo",
        type=float,
        default=110,
    )

    parser.add_argument(
        "--program",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--time-signature",
        type=str,
        default="4/4",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--primer",
        type=Path,
        default=Path("nofile"),
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    tokenizer = load_tokenizer(args.tokenizer_config)

    model, config = load_model(
        args.checkpoint,
        device,
    )

    context_length = config["context_length"]

    if len(tokenizer) != config["vocab_size"]:
        raise ValueError(
            "Tokenizer vocabulary size does not match the model. "
            f"Tokenizer: {len(tokenizer)}, "
            f"model: {config['vocab_size']}"
        )

    initial_ids = create_initial_sequence(
        tokenizer,
        tempo=args.tempo,
        program=args.program,
        time_signature=args.time_signature,
    )

    if args.primer != Path("nofile"):
        primer_ids = create_primer_from_midi(
            tokenizer,
            args.primer,
            n_bars=4,
        )
        initial_ids = primer_ids

    print("Initial prefix:")
    for token in initial_ids:
        print(f"  {token}", end="")

    if args.decoding == "random":
        generated_ids = generate_random(
            model=model,
            tokenizer=tokenizer,
            initial_ids=initial_ids,
            max_new_tokens=args.max_new_tokens,
            context_length=context_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
    else:
        generated_ids = generate_beam_search(
            model=model,
            tokenizer=tokenizer,
            initial_ids=initial_ids,
            max_new_tokens=args.max_new_tokens,
            context_length=context_length,
            beam_width=args.beam_width,
            device=device,
        )

    _, error_ratio = save_midi(
        tokenizer,
        generated_ids,
        args.output,
    )

    token_output = args.output.with_suffix(".json")

    metadata = {
        "ids": generated_ids,
        "decoding": args.decoding,
        "context_length": context_length,
        "seed": args.seed,
        "token_error_ratio": error_ratio,
    }

    if args.decoding == "random":
        metadata["temperature"] = args.temperature
        metadata["top_k"] = args.top_k
    else:
        metadata["beam_width"] = args.beam_width

    with token_output.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Token sequence saved to: {token_output}")


if __name__ == "__main__":
    main()