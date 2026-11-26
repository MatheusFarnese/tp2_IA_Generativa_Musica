import numpy as np
from compare_musical_metric_distributions import compare_metric

def js_from_samples(
    original,
    generated,
    value_range,
    n_bins=10,
    alpha=0.5,
):
    """Calculate histogram-based Jensen-Shannon divergence."""

    comparison = compare_metric(
        original_values=original,
        generated_values=generated,
        value_range=value_range,
        n_bins=n_bins,
        alpha=alpha,
    )

    return comparison["js_divergence"]


def permutation_test(
    original,
    generated,
    value_range,
    n_bins=10,
    alpha=0.5,
    n_permutations=10000,
    seed=42,
):
    """
    Test whether original and generated samples
    come from the same distribution.
    """

    rng = np.random.default_rng(seed)

    original = np.asarray(original, dtype=float)
    generated = np.asarray(generated, dtype=float)

    observed = js_from_samples(
        original,
        generated,
        value_range,
        n_bins,
        alpha,
    )

    pooled = np.concatenate([original, generated])

    n_original = len(original)
    count = 0

    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)

        perm_original = shuffled[:n_original]
        perm_generated = shuffled[n_original:]

        perm_js = js_from_samples(
            perm_original,
            perm_generated,
            value_range,
            n_bins,
            alpha,
        )

        if perm_js >= observed:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return {
        "observed_js": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def bootstrap_js_ci(
    original,
    generated,
    value_range,
    n_bins=10,
    alpha=0.5,
    n_bootstrap=10000,
    confidence=0.95,
    seed=42,
):
    """
    Estimate a confidence interval for the JSD
    by independently resampling MIDI compositions.
    """

    rng = np.random.default_rng(seed)

    original = np.asarray(original, dtype=float)
    generated = np.asarray(generated, dtype=float)

    bootstrap_values = []

    for _ in range(n_bootstrap):
        boot_original = rng.choice(
            original,
            size=len(original),
            replace=True,
        )

        boot_generated = rng.choice(
            generated,
            size=len(generated),
            replace=True,
        )

        value = js_from_samples(
            boot_original,
            boot_generated,
            value_range,
            n_bins,
            alpha,
        )

        bootstrap_values.append(value)

    tail = (1 - confidence) / 2

    lower, upper = np.quantile(
        bootstrap_values,
        [tail, 1 - tail],
    )

    return {
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
    }


def compare_two_jsds_bootstrap(
    reference,
    generated_a,
    generated_b,
    value_range,
    n_bins=10,
    alpha=0.5,
    n_bootstrap=10000,
    confidence=0.95,
    seed=42,
):
    """
    Compare the JSDs of two generators against the
    same reference dataset.

    Assumes independent generated samples from the
    two models. The reference sample is shared in
    every bootstrap replicate.
    """

    rng = np.random.default_rng(seed)

    reference = np.asarray(reference, dtype=float)
    generated_a = np.asarray(generated_a, dtype=float)
    generated_b = np.asarray(generated_b, dtype=float)

    differences = []

    for _ in range(n_bootstrap):
        boot_reference = rng.choice(
            reference,
            size=len(reference),
            replace=True,
        )

        boot_a = rng.choice(
            generated_a,
            size=len(generated_a),
            replace=True,
        )

        boot_b = rng.choice(
            generated_b,
            size=len(generated_b),
            replace=True,
        )

        js_a = js_from_samples(
            boot_reference,
            boot_a,
            value_range,
            n_bins,
            alpha,
        )

        js_b = js_from_samples(
            boot_reference,
            boot_b,
            value_range,
            n_bins,
            alpha,
        )

        differences.append(js_a - js_b)

    tail = (1 - confidence) / 2

    lower, upper = np.quantile(
        differences,
        [tail, 1 - tail],
    )

    return {
        "mean_difference": float(np.mean(differences)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
        "significantly_different": (
            lower > 0 or upper < 0
        ),
    }


import csv
import math
import numpy as np

# --------------------------------------------------
# Load individual metric values from CSV
# --------------------------------------------------

def load_metric(csv_path, dataset, metric):
    values = {}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["dataset"] != dataset:
                continue

            raw_value = row[metric]

            if not raw_value:
                continue

            value = float(raw_value)

            if math.isfinite(value):
                values[row["file"]] = value

    return values


# --------------------------------------------------
# Load reference and generated distributions
# --------------------------------------------------

csv_128 = "metric_distribution/128/per_file_metrics.csv"
csv_256 = "metric_distribution/256/per_file_metrics.csv"

# Metric ranges
metric_ranges = {
    "pitch_class_entropy": (0.0, math.log2(12)),
    "scale_consistency": (0.0, 1.0),
    "groove_consistency": (0.0, 1.0),
}

# Bootstrap configuration
N_BOOTSTRAP = 10000
N_BINS = 10
ALPHA = 0.5
SEED = 42


for metric, value_range in metric_ranges.items():

    # Original reference distribution
    original_128 = load_metric(
        csv_128, "original", metric
    )

    original_256 = load_metric(
        csv_256, "original", metric
    )

    # Verify that both experiments use the same reference.
    if original_128.keys() != original_256.keys():
        raise ValueError(
            f"Different reference files for {metric}."
        )

    for name in original_128:
        if not math.isclose(
            original_128[name],
            original_256[name],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"Reference metric mismatch for {name}."
            )

    reference = [
        original_128[name]
        for name in sorted(original_128)
    ]

    # Generated samples from both models.
    generated_128 = list(
        load_metric(csv_128, "generated", metric).values()
    )

    generated_256 = list(
        load_metric(csv_256, "generated", metric).values()
    )

    if not reference or not generated_128 or not generated_256:
        print(f"Skipping {metric}: insufficient observations.")
        continue

    # --------------------------------------------------
    # Calculate the observed JSDs
    # --------------------------------------------------

    js_128 = js_from_samples(
        reference,
        generated_128,
        value_range,
        N_BINS,
        ALPHA,
    )

    js_256 = js_from_samples(
        reference,
        generated_256,
        value_range,
        N_BINS,
        ALPHA,
    )

    observed_difference = js_128 - js_256

    # --------------------------------------------------
    # Compare divergences using bootstrap
    # --------------------------------------------------

    result = compare_two_jsds_bootstrap(
        reference=reference,
        generated_a=generated_128,
        generated_b=generated_256,
        value_range=value_range,
        n_bins=N_BINS,
        alpha=ALPHA,
        n_bootstrap=N_BOOTSTRAP,
        confidence=0.95,
        seed=SEED,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print(metric.upper())
    print("=" * 65)

    print(f"Original compositions: {len(reference)}")
    print(f"GRU-128 compositions:  {len(generated_128)}")
    print(f"GRU-256 compositions:  {len(generated_256)}")

    print(f"\nJSD {csv_128}: {js_128:.6f} bits")
    print(f"JSD {csv_256}: {js_256:.6f} bits")

    print(
        f"\nObserved difference (128 - 256): "
        f"{observed_difference:.6f}"
    )

    print(
        f"Bootstrap 95% CI: "
        f"[{result['ci_lower']:.6f}, "
        f"{result['ci_upper']:.6f}]"
    )

    if result["ci_lower"] > 0:
        print(
            "Conclusion: GRU-256 has significantly "
            "lower divergence."
        )

    elif result["ci_upper"] < 0:
        print(
            "Conclusion: GRU-128 has significantly "
            "lower divergence."
        )

    else:
        print(
            "Conclusion: No statistically significant "
            "difference detected."
        )
