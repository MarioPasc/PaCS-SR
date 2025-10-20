# Metrics Analysis CLI

## Overview

The `pacs-sr-analyze-metrics` command provides end-to-end metrics computation and statistical analysis for volumetric predictions. It compares super-resolution predictions against ground truth volumes and optionally generates visualization plots.

## Installation

After installing the package, the command is available as:
```bash
pacs-sr-analyze-metrics --config <config_file.yaml>
```

To enable visualization features, install with the `viz` extras:
```bash
pip install -e ".[viz]"
```

## Usage

### Basic Usage

Compute metrics and perform statistical analysis:
```bash
pacs-sr-analyze-metrics --config configs/analysis_config.yaml
```

### With Visualization

Compute metrics and generate plots:
```bash
pacs-sr-analyze-metrics --config configs/analysis_config.yaml --visualize
```

### Custom Output

Specify custom output directory for plots:
```bash
pacs-sr-analyze-metrics --config configs/analysis_config.yaml \
    --visualize \
    --plot-out results/figures
```

### Using Pre-computed Stats

If you already have stats NPZ files and want to generate plots only:
```bash
pacs-sr-analyze-metrics --config configs/analysis_config.yaml \
    --visualize \
    --stats-npz results/analysis/sr_stats_summary.npz \
    --metrics-npz results/analysis/metrics.npz \
    --plot-out results/figures
```

## Configuration

The analysis requires a YAML configuration file with the following sections:

### 1. Data Configuration (`analysis.data`)

Specifies input data paths and format:
- `gt_dir`: Ground truth (high-resolution) volumes directory
- `pred_dirs`: Dictionary mapping method names to prediction directories
- `mask_dir`: Optional brain mask directory
- `sequences`: List of MRI sequences to analyze (e.g., T1C, T2W)
- `file_ext`: File extension (.nii.gz, .nii, .npy, .npz)

### 2. Metrics Configuration (`analysis.metrics`)

Defines which metrics to compute:
- `compute`: List of metrics (psnr, ssim, mae, rmse, ncc)
- `ssim`: SSIM-specific parameters (window size, sigma, etc.)
- `psnr`: PSNR-specific parameters (data range)
- `crop_border`: Border pixels to exclude

### 3. Statistics Configuration (`analysis.stats`)

Defines statistical tests and corrections:
- `tests`: Statistical tests to perform (ttest_paired, wilcoxon)
- `effect_sizes`: Effect size measures (cohens_dz, cliffs_delta)
- `multiple_comparison`: Correction method (none, bonferroni, fdr_bh)
- `bootstrap`: Bootstrap settings for confidence intervals

### 4. I/O Configuration (`analysis.io`)

Output settings:
- `output_dir`: Where to save results
- `write_csv`: Save results as CSV
- `write_json`: Save results as JSON

### 5. Runtime Configuration (`analysis.runtime`)

Computational settings:
- `num_workers`: Number of parallel processes
- `seed`: Random seed for reproducibility
- `chunk_voxels`: Memory control parameter

## Output Files

The analysis generates the following outputs in `output_dir`:

### Metrics Files
- `metrics_per_sample.csv`: Per-case metrics for all methods
- `metrics_per_sample.json`: JSON format of per-case metrics
- `metrics_summary.csv`: Aggregated statistics (mean, std, median)

### Statistical Analysis Files
- `stats_pairwise.csv`: Pairwise comparison results with p-values
- `stats_pairwise.json`: JSON format of pairwise comparisons
- `errors.log`: Any errors encountered during processing (if any)

### Visualization Files (if --visualize is used)
- `sr_panel_<pulse>_1x4.pdf/png`: Combined analysis panel per pulse
  - Forest plot (effect sizes)
  - Wilcoxon heatmap (p-values)
  - Adjusted means plot
  - ICC stability bars

## Example Workflow

1. **Prepare your data structure:**
   ```
   data/
   ├── ground_truth/
   │   ├── case001_T1C.nii.gz
   │   ├── case001_T2W.nii.gz
   │   └── ...
   ├── bspline_predictions/
   │   ├── case001_T1C.nii.gz
   │   └── ...
   └── pacssr_predictions/
       ├── case001_T1C.nii.gz
       └── ...
   ```

2. **Configure analysis:** Edit `configs/analysis_config.yaml` with your paths

3. **Run analysis:**
   ```bash
   pacs-sr-analyze-metrics --config configs/analysis_config.yaml
   ```

4. **Generate visualizations:**
   ```bash
   pacs-sr-analyze-metrics --config configs/analysis_config.yaml --visualize
   ```

5. **Review results** in the `output_dir` specified in your config

## Requirements

### Core Requirements (always needed)
- numpy >= 1.23
- scipy >= 1.10
- pandas >= 1.5
- scikit-image >= 0.21
- nibabel >= 5.0 (for NIfTI files)
- pydantic >= 2.0

### Visualization Requirements (optional, for --visualize)
- matplotlib >= 3.6
- statsmodels >= 0.14

Install with: `pip install -e ".[viz]"`

## Notes

- All predictions must have the same filename structure as ground truth
- Missing files can be allowed with `allow_missing: true` in config
- For large datasets, adjust `num_workers` and `chunk_voxels` to control memory usage
- Statistical tests assume paired data by default (same cases across methods)
- Multiple comparison correction helps control false discovery rate

## Troubleshooting

### "No metrics computed. Check configuration and paths."
- Verify that file paths in config are correct
- Check that filenames match the pattern `{case}_{seq}{ext}`
- Ensure ground truth and prediction files exist

### Memory errors
- Reduce `chunk_voxels` in config
- Reduce `num_workers`
- Process sequences separately

### Visualization fails
- Ensure visualization dependencies are installed: `pip install -e ".[viz]"`
- Check that stats NPZ file exists in output directory
- Verify matplotlib backend is properly configured

## See Also

- Configuration example: `configs/analysis_config.yaml`
- Stats visualization: `pacs_sr/utils/stats.py`
- Analysis implementation: `pacs_sr/analysis/analyze_metrics.py`
