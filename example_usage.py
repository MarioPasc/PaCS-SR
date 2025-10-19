"""
Example script demonstrating PaCS-SR usage with enhanced logging and NPZ weight maps.

This script shows:
1. Loading configuration
2. Training with real-time logging
3. Evaluation with metrics tracking
4. Loading and analyzing weight maps
"""

from pathlib import Path
import json
import numpy as np

from pacs_sr.config.config import load_config
from pacs_sr.model.model import PatchwiseConvexStacker
from pacs_sr.utils.weight_maps import (
    load_weight_maps_npz,
    analyze_weight_statistics,
    compute_weight_entropy,
    get_dominant_model_map
)


def main():
    """Main example workflow."""

    # 1. Load configuration
    print("Loading configuration...")
    config_path = Path("configs/config.yaml")
    config = load_config(config_path)

    # 2. Load K-fold manifest
    # This should be created using pacs_sr.data.folds_builder
    print("Loading K-fold manifest...")
    with open(config.cv_json, "r") as f:
        manifest = json.load(f)

    # 3. Initialize model
    # Logger is automatically created based on config settings
    print("Initializing model...")
    model = PatchwiseConvexStacker(config)

    # 4. Train on first spacing/pulse combination
    spacing = config.spacings[0]  # e.g., "3mm"
    pulse = config.pulses[0]      # e.g., "t1c"

    print(f"\nTraining for {spacing} {pulse}...")
    print("=" * 80)

    # This will show:
    # - Session header with timestamp
    # - Configuration parameters
    # - Training progress for each patient
    # - Region optimization progress
    # - Training completion summary
    weights = model.fit_one(manifest, spacing, pulse)

    print(f"\nLearned {len(weights)} regional weight vectors")

    # 5. Evaluate on train and test sets
    print(f"\nEvaluating {spacing} {pulse}...")
    print("=" * 80)

    # This will show:
    # - Evaluation progress for train set
    # - Per-patient metrics
    # - Aggregate train metrics
    # - Evaluation progress for test set
    # - Per-patient metrics
    # - Aggregate test metrics
    # - File saving notifications
    results = model.evaluate_split(manifest, spacing, pulse)

    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print("=" * 80)
    print(f"Train Set:")
    for metric, value in results["train"].items():
        print(f"  {metric.upper():10s}: {value:.6f}")
    print(f"\nTest Set:")
    for metric, value in results["test"].items():
        print(f"  {metric.upper():10s}: {value:.6f}")

    # 6. Load and analyze weight maps (if saved)
    if config.save_weight_volumes:
        print("\n" + "=" * 80)
        print("WEIGHT MAP ANALYSIS:")
        print("=" * 80)

        # Get first test patient
        test_patients = list(manifest["test"].keys())
        if test_patients:
            first_patient = test_patients[0]

            # Load weight maps
            out_dir = Path(config.out_root) / config.experiment_name / spacing / pulse
            weight_path = out_dir / f"{first_patient}_weights_test.npz"

            if weight_path.exists():
                print(f"\nLoading weight maps for {first_patient}...")
                data = load_weight_maps_npz(weight_path)

                weight_maps = data["weight_maps"]
                model_names = data["model_names"]

                print(f"Weight map shape: {weight_maps.shape}")
                print(f"Models: {', '.join(model_names)}")

                # Compute statistics
                stats = analyze_weight_statistics(weight_maps, model_names)
                print("\nPer-Model Weight Statistics:")
                for model_name, model_stats in stats.items():
                    print(f"\n  {model_name}:")
                    print(f"    Mean:   {model_stats['mean']:.4f}")
                    print(f"    Std:    {model_stats['std']:.4f}")
                    print(f"    Median: {model_stats['median']:.4f}")
                    print(f"    Range:  [{model_stats['min']:.4f}, {model_stats['max']:.4f}]")

                # Compute entropy
                entropy = compute_weight_entropy(weight_maps)
                print(f"\nWeight Entropy Statistics:")
                print(f"  Mean entropy:   {np.mean(entropy):.4f}")
                print(f"  Median entropy: {np.median(entropy):.4f}")
                print(f"  Max entropy:    {np.max(entropy):.4f} (uniform blending)")
                print(f"  Min entropy:    {np.min(entropy):.4f} (single model)")

                # Get dominant models
                dominant = get_dominant_model_map(weight_maps)
                print(f"\nDominant Model Distribution:")
                for idx, model_name in enumerate(model_names):
                    count = np.sum(dominant == idx)
                    percentage = count / dominant.size * 100
                    print(f"  {model_name:10s}: {count:8d} voxels ({percentage:5.2f}%)")
            else:
                print(f"Weight map file not found: {weight_path}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)

    # Log session end
    model.logger.log_session_end()


def analyze_saved_weights(weight_npz_path: Path):
    """
    Standalone function to analyze saved weight maps.

    Args:
        weight_npz_path: Path to NPZ weight map file
    """
    print(f"Analyzing weight maps: {weight_npz_path}")

    # Load data
    data = load_weight_maps_npz(weight_npz_path)

    weight_maps = data["weight_maps"]
    model_names = data["model_names"]
    patient_id = data["patient_id"]
    spacing = data["spacing"]
    pulse = data["pulse"]

    print(f"\nPatient: {patient_id}")
    print(f"Spacing: {spacing}")
    print(f"Pulse: {pulse}")
    print(f"Volume shape: {data['shape']}")
    print(f"Models: {', '.join(model_names)}")

    # Statistics
    stats = analyze_weight_statistics(weight_maps, model_names)
    print("\nWeight Statistics by Model:")
    print("-" * 60)
    print(f"{'Model':<15} {'Mean':<10} {'Std':<10} {'Median':<10}")
    print("-" * 60)
    for model_name, model_stats in stats.items():
        print(f"{model_name:<15} {model_stats['mean']:<10.4f} "
              f"{model_stats['std']:<10.4f} {model_stats['median']:<10.4f}")

    # Entropy analysis
    entropy = compute_weight_entropy(weight_maps)
    print(f"\nEntropy Analysis:")
    print(f"  Mean:   {np.mean(entropy):.4f}")
    print(f"  Median: {np.median(entropy):.4f}")
    print(f"  Q25:    {np.percentile(entropy, 25):.4f}")
    print(f"  Q75:    {np.percentile(entropy, 75):.4f}")

    # Dominant model
    dominant = get_dominant_model_map(weight_maps)
    print(f"\nDominant Model Frequency:")
    for idx, model_name in enumerate(model_names):
        count = np.sum(dominant == idx)
        pct = count / dominant.size * 100
        print(f"  {model_name:<15}: {pct:6.2f}%")


if __name__ == "__main__":
    # Run main example
    main()

    # Example: Analyze a specific weight map file
    # analyze_saved_weights(Path("output/experiment/3mm/t1c/patient001_weights_test.npz"))
