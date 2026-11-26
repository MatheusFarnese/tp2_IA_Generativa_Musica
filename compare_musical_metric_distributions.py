
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import muspy


# Theoretical ranges of the metrics.
METRIC_RANGES = {
    "pitch_class_entropy": (0.0, math.log2(12)),
    "scale_consistency": (0.0, 1.0),
    "groove_consistency": (0.0, 1.0),
}


# ============================================================
# MIDI evaluation
# ============================================================

def get_measure_resolution(music, default_time_signature):
    """
    Calculate the number of time steps per measure.

    Groove consistency requires a constant time signature.
    MusPy resolution is expressed in ticks per quarter note.
    """
    time_signatures = music.time_signatures

    if not time_signatures:
        numerator, denominator = default_time_signature
    else:
        signatures = {
            (ts.numerator, ts.denominator)
            for ts in time_signatures
        }

        if len(signatures) != 1:
            raise ValueError(
                "Groove consistency requires a constant "
                f"time signature. Found: {sorted(signatures)}"
            )

        numerator, denominator = next(iter(signatures))

    resolution = (
        music.resolution * numerator * 4 / denominator
    )

    rounded = round(resolution)

    if not math.isclose(
        resolution,
        rounded,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"Non-integer measure resolution: {resolution}"
        )

    return int(rounded)


def safe_metric(value):
    """Convert a metric value into a finite Python float or None."""
    value = float(value)

    if not math.isfinite(value):
        return None

    return value


def evaluate_midi(midi_path, backend, default_time_signature):
    """Calculate all three musical metrics for one MIDI file."""

    music = muspy.read_midi(
        midi_path,
        backend=backend,
    )

    result = {
        "file": midi_path.name,
        "pitch_class_entropy": safe_metric(
            muspy.pitch_class_entropy(music)
        ),
        "scale_consistency": safe_metric(
            muspy.scale_consistency(music)
        ),
        "groove_consistency": None,
        "groove_status": "",
    }

    try:
        measure_resolution = get_measure_resolution(
            music,
            default_time_signature,
        )

        groove = muspy.groove_consistency(
            music,
            measure_resolution=measure_resolution,
        )

        result["groove_consistency"] = safe_metric(groove)

        if result["groove_consistency"] is None:
            result["groove_status"] = (
                "Undefined: fewer than two measures or no valid onsets"
            )
        else:
            result["groove_status"] = "OK"

    except Exception as exc:
        result["groove_status"] = str(exc)

    return result


def evaluate_folder(folder, backend, default_time_signature):
    """Evaluate every MIDI file in a folder recursively."""

    midi_files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".mid", ".midi"}
    )

    if not midi_files:
        raise RuntimeError(f"No MIDI files found in {folder}")

    results = []
    errors = []

    print(f"\nEvaluating folder: {folder}")
    print(f"Found {len(midi_files)} MIDI files.\n")

    for index, midi_path in enumerate(midi_files, start=1):
        #print(f"[{index}/{len(midi_files)}] {midi_path.name}")

        try:
            result = evaluate_midi(
                midi_path,
                backend,
                default_time_signature,
            )

            results.append(result)

            for metric in METRIC_RANGES:
                value = result[metric]

                formatted = (
                    f"{value:.4f}" if value is not None else "N/A"
                )

                #print(f"  {metric}: {formatted}")

            #if result["groove_status"] != "OK":
                #print(
                    #f"  Groove note: {result['groove_status']}"
                #)

        except Exception as exc:
            errors.append({
                "file": str(midi_path),
                "error": str(exc),
            })

            print(f"  ERROR: {exc}")

    return results, errors


# ============================================================
# Distribution estimation
# ============================================================

def make_histogram(values, value_range, n_bins):
    """
    Estimate a probability distribution using a fixed-width histogram.
    """
    lower, upper = value_range
    counts = [0] * n_bins

    for value in values:
        if value < lower - 1e-8 or value > upper + 1e-8:
            raise ValueError(
                f"Metric value {value} is outside [{lower}, {upper}]."
            )

        # Clamp minor numerical deviations.
        value = min(max(value, lower), upper)

        index = int(
            (value - lower) / (upper - lower) * n_bins
        )

        # The upper boundary belongs to the final bin.
        index = min(index, n_bins - 1)

        counts[index] += 1

    return counts


def smooth_distribution(counts, alpha):
    """
    Additive smoothing to avoid zero-probability bins.
    """
    total = sum(counts) + alpha * len(counts)

    return [
        (count + alpha) / total
        for count in counts
    ]


# ============================================================
# Divergence calculations
# ============================================================

def kl_divergence(p, q):
    """Calculate KL(P || Q), in bits."""
    return sum(
        pi * math.log2(pi / qi)
        for pi, qi in zip(p, q)
        if pi > 0
    )


def js_divergence(p, q):
    """Calculate symmetric Jensen-Shannon divergence, in bits."""
    midpoint = [
        (pi + qi) / 2
        for pi, qi in zip(p, q)
    ]

    return (
        kl_divergence(p, midpoint)
        + kl_divergence(q, midpoint)
    ) / 2


def compare_metric(
    original_values,
    generated_values,
    value_range,
    n_bins,
    alpha,
):
    """Compare the empirical distributions of one metric."""

    original_counts = make_histogram(
        original_values,
        value_range,
        n_bins,
    )

    generated_counts = make_histogram(
        generated_values,
        value_range,
        n_bins,
    )

    p = smooth_distribution(original_counts, alpha)
    q = smooth_distribution(generated_counts, alpha)

    return {
        "original_counts": original_counts,
        "generated_counts": generated_counts,
        "original_distribution": p,
        "generated_distribution": q,
        "kl_original_generated": kl_divergence(p, q),
        "kl_generated_original": kl_divergence(q, p),
        "js_divergence": js_divergence(p, q),
    }


# ============================================================
# Output
# ============================================================

def calculate_summary(values):
    """Calculate basic descriptive statistics."""
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def save_per_file_csv(original, generated, output_path):
    """Save metric values for all MIDI files."""

    rows = []

    for dataset_name, results in [
        ("original", original),
        ("generated", generated),
    ]:
        for result in results:
            rows.append({
                "dataset": dataset_name,
                **result,
            })

    fieldnames = [
        "dataset",
        "file",
        "pitch_class_entropy",
        "scale_consistency",
        "groove_consistency",
        "groove_status",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare distributions of Pitch Class Entropy, "
            "Scale Consistency and Groove Consistency "
            "between two MIDI folders."
        )
    )

    parser.add_argument(
        "original_folder",
        type=Path,
        help="Folder containing original MIDI files.",
    )

    parser.add_argument(
        "generated_folder",
        type=Path,
        help="Folder containing generated MIDI files.",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of histogram bins per metric. Default: 10.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Additive smoothing parameter. Default: 0.5.",
    )

    parser.add_argument(
        "--time-signature",
        default="4/4",
        help="Default time signature when absent from MIDI. Default: 4/4.",
    )

    parser.add_argument(
        "--backend",
        choices=["mido", "pretty_midi"],
        default="mido",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metric_distribution_results"),
    )

    args = parser.parse_args()

    if not args.original_folder.is_dir():
        parser.error(
            f"Original folder does not exist: {args.original_folder}"
        )

    if not args.generated_folder.is_dir():
        parser.error(
            f"Generated folder does not exist: {args.generated_folder}"
        )

    if args.bins < 2:
        parser.error("--bins must be at least 2.")

    if args.alpha <= 0:
        parser.error("--alpha must be greater than zero.")

    try:
        numerator, denominator = map(
            int,
            args.time_signature.split("/"),
        )

        if numerator <= 0 or denominator <= 0:
            raise ValueError

        default_time_signature = (numerator, denominator)

    except ValueError:
        parser.error(
            "--time-signature must have the format numerator/denominator."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate original and generated folders.
    original_results, original_errors = evaluate_folder(
        args.original_folder,
        args.backend,
        default_time_signature,
    )

    generated_results, generated_errors = evaluate_folder(
        args.generated_folder,
        args.backend,
        default_time_signature,
    )

    # Save individual metrics.
    save_per_file_csv(
        original_results,
        generated_results,
        args.output_dir / "per_file_metrics.csv",
    )

    summary = {
        "original_folder": str(args.original_folder.resolve()),
        "generated_folder": str(args.generated_folder.resolve()),
        "bins": args.bins,
        "alpha": args.alpha,
        "time_signature_default": args.time_signature,
        "metrics": {},
        "errors": {
            "original": original_errors,
            "generated": generated_errors,
        },
    }

    distribution_rows = []

    print("\n" + "=" * 75)
    print("DISTRIBUTION COMPARISON")
    print("=" * 75)

    for metric, value_range in METRIC_RANGES.items():

        original_values = [
            row[metric]
            for row in original_results
            if row[metric] is not None
        ]

        generated_values = [
            row[metric]
            for row in generated_results
            if row[metric] is not None
        ]

        if not original_values or not generated_values:
            print(f"\n{metric}: insufficient valid observations.")
            continue

        comparison = compare_metric(
            original_values,
            generated_values,
            value_range,
            args.bins,
            args.alpha,
        )

        summary["metrics"][metric] = {
            "original_statistics": calculate_summary(
                original_values
            ),
            "generated_statistics": calculate_summary(
                generated_values
            ),
            "kl_original_generated_bits":
                comparison["kl_original_generated"],
            "kl_generated_original_bits":
                comparison["kl_generated_original"],
            "js_divergence_bits":
                comparison["js_divergence"],
        }

        print(f"\n{metric.upper()}")
        print("-" * 65)

        print(
            f"Original:  n={len(original_values):4d}, "
            f"mean={mean(original_values):.4f}"
        )

        print(
            f"Generated: n={len(generated_values):4d}, "
            f"mean={mean(generated_values):.4f}"
        )

        print(
            f"KL (original || generated): "
            f"{comparison['kl_original_generated']:.6f} bits"
        )

        print(
            f"KL (generated || original): "
            f"{comparison['kl_generated_original']:.6f} bits"
        )

        print(
            f"Jensen-Shannon divergence:  "
            f"{comparison['js_divergence']:.6f} bits"
        )

        # Save histogram probabilities.
        lower, upper = value_range
        bin_width = (upper - lower) / args.bins

        for i in range(args.bins):
            distribution_rows.append({
                "metric": metric,
                "bin": i,
                "lower_bound": lower + i * bin_width,
                "upper_bound": lower + (i + 1) * bin_width,
                "original_count": comparison["original_counts"][i],
                "generated_count": comparison["generated_counts"][i],
                "original_probability":
                    comparison["original_distribution"][i],
                "generated_probability":
                    comparison["generated_distribution"][i],
            })

    # Save histogram distributions.
    with (
        args.output_dir / "metric_distributions.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        if distribution_rows:
            writer = csv.DictWriter(
                f,
                fieldnames=list(distribution_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(distribution_rows)

    # Save summary.
    with (
        args.output_dir / "divergence_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 75)
    print("RESULTS SAVED")
    print("=" * 75)
    print(f"Per-file metrics: {args.output_dir / 'per_file_metrics.csv'}")
    print(
        f"Distributions:    "
        f"{args.output_dir / 'metric_distributions.csv'}"
    )
    print(
        f"Summary:          "
        f"{args.output_dir / 'divergence_summary.json'}"
    )


if __name__ == "__main__":
    main()
