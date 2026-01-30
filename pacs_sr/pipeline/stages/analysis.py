"""
Analysis Stage
==============

Aggregates metrics across folds and runs statistical comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, List, Optional

import numpy as np

from .base import PipelineStage, StageResult

if TYPE_CHECKING:
    from pacs_sr.pipeline.checkpoint import CheckpointManager
    from pacs_sr.pipeline.context import PipelineContext


class AnalysisStage(PipelineStage):
    """
    Analysis stage: Aggregate metrics and run statistical tests.

    Tasks:
    - Collect metrics from all training tasks
    - Aggregate per-fold and overall statistics
    - Compare PaCS-SR vs individual experts
    - Run statistical significance tests
    - Generate analysis CSV and JSON outputs
    """

    @property
    def name(self) -> str:
        return "analysis"

    def run(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> StageResult:
        """Execute analysis stage."""
        self.log(context, "Starting analysis stage...")

        # Collect all metrics
        self.log(context, "Collecting metrics from training tasks...")
        all_metrics = self._collect_metrics(context, checkpoint)

        if not all_metrics:
            return StageResult.fail("No metrics found from training tasks")

        # Aggregate metrics
        self.log(context, "Aggregating metrics...")
        aggregated = self._aggregate_metrics(context, all_metrics)

        # Save aggregated metrics
        self._save_aggregated_metrics(context, aggregated)

        # Compare with experts (if expert metrics available)
        self.log(context, "Comparing PaCS-SR vs experts...")
        comparison = self._compare_with_experts(context, aggregated)

        if comparison:
            self._save_comparison(context, comparison)

        # Run statistical tests
        self.log(context, "Running statistical tests...")
        stats = self._run_statistical_tests(context, all_metrics)

        if stats:
            self._save_statistical_tests(context, stats)

        # Run regional specialization analysis (for explainability)
        self.log(context, "Running regional specialization analysis...")
        regional_results = self._run_regional_analysis(context)
        if regional_results:
            self._save_regional_analysis(context, regional_results)

        self.log(context, "Analysis stage completed successfully")
        return StageResult.ok("Analysis completed", data={"aggregated": aggregated})

    def _collect_metrics(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> Dict[str, Dict[str, Any]]:
        """Collect metrics from all completed training tasks."""
        metrics = {}

        for fold, spacing, pulse in context.get_training_tasks():
            task_id = context.task_id(fold, spacing, pulse)

            # Get metrics from checkpoint
            task_metrics = checkpoint.get_task_metrics(task_id)
            if task_metrics:
                metrics[task_id] = {
                    "fold": fold,
                    "spacing": spacing,
                    "pulse": pulse,
                    "metrics": task_metrics,
                }

            # Also try to load from metrics.json file
            metrics_path = (
                context.training_dir / spacing / "model_data" / f"fold_{fold}" / pulse / "metrics.json"
            )
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r") as f:
                        file_metrics = json.load(f)
                    if task_id not in metrics:
                        metrics[task_id] = {
                            "fold": fold,
                            "spacing": spacing,
                            "pulse": pulse,
                            "metrics": file_metrics,
                        }
                    else:
                        # Merge with checkpoint metrics
                        metrics[task_id]["metrics"].update(file_metrics)
                except Exception:
                    pass

        return metrics

    def _aggregate_metrics(
        self,
        context: "PipelineContext",
        all_metrics: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate metrics across folds."""
        aggregated = {
            "per_fold": {},
            "per_spacing": {},
            "per_pulse": {},
            "overall": {},
        }

        # Group by different dimensions
        by_spacing_pulse = {}
        by_spacing = {}
        by_pulse = {}

        for task_id, data in all_metrics.items():
            fold = data["fold"]
            spacing = data["spacing"]
            pulse = data["pulse"]
            test_metrics = data["metrics"].get("test", {})

            # Store per-fold data
            aggregated["per_fold"][task_id] = {
                "fold": fold,
                "spacing": spacing,
                "pulse": pulse,
                **test_metrics,
            }

            # Group by spacing/pulse
            key = f"{spacing}_{pulse}"
            if key not in by_spacing_pulse:
                by_spacing_pulse[key] = []
            by_spacing_pulse[key].append(test_metrics)

            # Group by spacing
            if spacing not in by_spacing:
                by_spacing[spacing] = []
            by_spacing[spacing].append(test_metrics)

            # Group by pulse
            if pulse not in by_pulse:
                by_pulse[pulse] = []
            by_pulse[pulse].append(test_metrics)

        # Compute aggregated statistics
        metric_names = ["psnr", "ssim", "mae", "mse"]

        # Per spacing/pulse combination
        for key, metrics_list in by_spacing_pulse.items():
            aggregated["per_spacing"][key] = self._compute_stats(metrics_list, metric_names)

        # Per spacing (across pulses)
        for spacing, metrics_list in by_spacing.items():
            aggregated["per_spacing"][spacing] = self._compute_stats(metrics_list, metric_names)

        # Per pulse (across spacings)
        for pulse, metrics_list in by_pulse.items():
            aggregated["per_pulse"][pulse] = self._compute_stats(metrics_list, metric_names)

        # Overall
        all_test_metrics = [
            data["metrics"].get("test", {})
            for data in all_metrics.values()
        ]
        aggregated["overall"] = self._compute_stats(all_test_metrics, metric_names)

        return aggregated

    def _compute_stats(
        self,
        metrics_list: List[Dict[str, float]],
        metric_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean, std, min, max for each metric."""
        stats = {}
        for name in metric_names:
            values = [m.get(name) for m in metrics_list if m.get(name) is not None]
            if values:
                stats[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n": len(values),
                }
        return stats

    def _save_aggregated_metrics(
        self,
        context: "PipelineContext",
        aggregated: Dict[str, Any],
    ) -> None:
        """Save aggregated metrics to files."""
        analysis_dir = context.analysis_dir
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = analysis_dir / "metrics_aggregated.json"
        with open(json_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        self.log(context, f"Saved aggregated metrics: {json_path}")

        # Save per-fold as CSV
        csv_path = analysis_dir / "metrics_per_fold.csv"
        self._save_per_fold_csv(context, aggregated["per_fold"], csv_path)
        self.log(context, f"Saved per-fold metrics: {csv_path}")

    def _save_per_fold_csv(
        self,
        context: "PipelineContext",
        per_fold: Dict[str, Dict],
        path: Path,
    ) -> None:
        """Save per-fold metrics as CSV."""
        import csv

        if not per_fold:
            return

        # Get all metric names
        sample = next(iter(per_fold.values()))
        metric_cols = [k for k in sample.keys() if k not in ["fold", "spacing", "pulse"]]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_id", "fold", "spacing", "pulse"] + metric_cols)

            for task_id, data in sorted(per_fold.items()):
                row = [
                    task_id,
                    data["fold"],
                    data["spacing"],
                    data["pulse"],
                ] + [data.get(m, "") for m in metric_cols]
                writer.writerow(row)

    def _compare_with_experts(
        self,
        context: "PipelineContext",
        aggregated: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Compare PaCS-SR metrics with individual expert models."""
        # Look for expert metrics in the data directory
        config = context.config
        models_root = Path(config.data.models_root)

        comparison = {
            "pacs_sr": aggregated.get("overall", {}),
            "experts": {},
        }

        # This is a placeholder - in practice, you'd load expert metrics
        # from their output directories
        for model in config.pacs_sr.models:
            model_dir = models_root / model
            if model_dir.exists():
                # TODO: Load expert metrics if available
                comparison["experts"][model] = {}

        return comparison if comparison["experts"] else None

    def _save_comparison(
        self,
        context: "PipelineContext",
        comparison: Dict[str, Any],
    ) -> None:
        """Save PaCS-SR vs experts comparison."""
        path = context.analysis_dir / "pacs_sr_vs_experts.json"
        with open(path, "w") as f:
            json.dump(comparison, f, indent=2)
        self.log(context, f"Saved comparison: {path}")

    def _run_statistical_tests(
        self,
        context: "PipelineContext",
        all_metrics: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Run statistical significance tests."""
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            self.log(context, "scipy not available, skipping statistical tests", "warning")
            return None

        results = {
            "tests": [],
            "summary": {},
        }

        # Group metrics by spacing for paired comparisons
        by_spacing = {}
        for task_id, data in all_metrics.items():
            spacing = data["spacing"]
            if spacing not in by_spacing:
                by_spacing[spacing] = []
            test_metrics = data["metrics"].get("test", {})
            if test_metrics:
                by_spacing[spacing].append(test_metrics)

        # Paired t-tests between spacings (if applicable)
        spacings = list(by_spacing.keys())
        for i, s1 in enumerate(spacings):
            for s2 in spacings[i+1:]:
                m1 = by_spacing[s1]
                m2 = by_spacing[s2]

                for metric in ["psnr", "ssim"]:
                    v1 = [m.get(metric) for m in m1 if m.get(metric)]
                    v2 = [m.get(metric) for m in m2 if m.get(metric)]

                    if len(v1) >= 2 and len(v2) >= 2:
                        # Independent samples t-test
                        t_stat, p_value = scipy_stats.ttest_ind(v1, v2)
                        results["tests"].append({
                            "test": "independent_t_test",
                            "metric": metric,
                            "groups": [s1, s2],
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant_0.05": bool(p_value < 0.05),
                        })

        # Normality tests
        all_psnr = []
        all_ssim = []
        for data in all_metrics.values():
            test_m = data["metrics"].get("test", {})
            if test_m.get("psnr"):
                all_psnr.append(test_m["psnr"])
            if test_m.get("ssim"):
                all_ssim.append(test_m["ssim"])

        if len(all_psnr) >= 3:
            stat, p = scipy_stats.shapiro(all_psnr)
            results["summary"]["psnr_normality"] = {
                "test": "shapiro_wilk",
                "statistic": float(stat),
                "p_value": float(p),
                "normal_0.05": bool(p > 0.05),
            }

        if len(all_ssim) >= 3:
            stat, p = scipy_stats.shapiro(all_ssim)
            results["summary"]["ssim_normality"] = {
                "test": "shapiro_wilk",
                "statistic": float(stat),
                "p_value": float(p),
                "normal_0.05": bool(p > 0.05),
            }

        return results

    def _save_statistical_tests(
        self,
        context: "PipelineContext",
        stats: Dict[str, Any],
    ) -> None:
        """Save statistical test results."""
        path = context.analysis_dir / "statistical_tests.json"
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        self.log(context, f"Saved statistical tests: {path}")

    def _run_regional_analysis(
        self,
        context: "PipelineContext",
    ) -> Optional[Dict[str, Any]]:
        """
        Run regional specialization analysis for explainability.

        This is key for the medical congress: it shows which SR experts
        excel in which brain regions, providing interpretable insights.
        """
        try:
            from pacs_sr.experiments.regional_specialization import (
                load_weight_maps,
                analyze_regional_weights,
                compute_specialization_index,
            )
        except ImportError:
            self.log(context, "Regional specialization module not available", "warning")
            return None

        # Find all weight NPZ files
        weight_files = list(context.training_dir.rglob("*_weights_*.npz"))

        if not weight_files:
            self.log(context, "No weight files found for regional analysis", "warning")
            return None

        self.log(context, f"Analyzing {len(weight_files)} weight files...")

        all_specialization = []
        global_stats_by_model = {}
        model_names = list(context.config.pacs_sr.models)

        for wf in weight_files:
            try:
                weight_maps, names, metadata = load_weight_maps(wf)
                results = analyze_regional_weights(weight_maps, names)
                all_specialization.append(results["specialization_index"])

                # Aggregate global stats
                for model, stats in results["global_stats"].items():
                    if model not in global_stats_by_model:
                        global_stats_by_model[model] = {"means": [], "stds": []}
                    global_stats_by_model[model]["means"].append(stats["mean"])
                    global_stats_by_model[model]["stds"].append(stats["std"])

            except Exception as e:
                self.log(context, f"Error processing {wf}: {e}", "warning")
                continue

        if not all_specialization:
            return None

        # Compute aggregated results
        results = {
            "n_files_analyzed": len(all_specialization),
            "specialization_index": {
                "mean": float(np.mean(all_specialization)),
                "std": float(np.std(all_specialization)),
                "min": float(np.min(all_specialization)),
                "max": float(np.max(all_specialization)),
            },
            "model_weights": {},
        }

        # Per-model weight statistics
        for model, stats in global_stats_by_model.items():
            results["model_weights"][model] = {
                "mean_weight": float(np.mean(stats["means"])),
                "std_weight": float(np.mean(stats["stds"])),
            }

        self.log(
            context,
            f"Regional analysis: Specialization Index = {results['specialization_index']['mean']:.3f} ± "
            f"{results['specialization_index']['std']:.3f}"
        )

        return results

    def _save_regional_analysis(
        self,
        context: "PipelineContext",
        results: Dict[str, Any],
    ) -> None:
        """Save regional specialization analysis results."""
        path = context.analysis_dir / "regional_specialization.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        self.log(context, f"Saved regional analysis: {path}")
