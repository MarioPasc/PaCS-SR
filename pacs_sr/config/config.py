from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union
import yaml
from pydantic import BaseModel, Field, field_validator, ValidationError

# -----------------------------
# Dataclass containers (lightweight, used at runtime)
# -----------------------------

@dataclass(frozen=True)
class NormConfig:
    kind: Literal["none", "zscore", "minmax"] = "none"
    clip: Optional[Tuple[float, float]] = None

@dataclass(frozen=True)
class SSIMConfig:
    win_size: int = 7
    gaussian_weights: bool = True
    sigma: float = 1.5
    use_sample_covariance: bool = False
    data_range: Union[Literal["auto"], float] = "auto"

@dataclass(frozen=True)
class PSNRConfig:
    data_range: Union[Literal["auto"], float] = "auto"

@dataclass(frozen=True)
class MetricsConfig:
    compute: Tuple[str, ...] = ("psnr", "ssim", "mae", "rmse", "ncc")
    crop_border: int = 0
    ssim: SSIMConfig = SSIMConfig()
    psnr: PSNRConfig = PSNRConfig()

@dataclass(frozen=True)
class AnalysisDataConfig:
    gt_dir: Path = Path("/")
    pred_dirs: Dict[str, Path] = field(default_factory=dict)
    mask_dir: Optional[Path] = None
    file_ext: Literal[".nii.gz", ".nii", ".npy", ".npz"] = ".nii.gz"
    sequences: Tuple[str, ...] = ("T1C", "T1N", "T2W", "T2F")
    cases_list: Optional[Path] = None
    filename_pattern: str = "{case}_{seq}"
    allow_missing: bool = False
    dtype: Literal["float32", "float64"] = "float32"
    norm: NormConfig = NormConfig()

@dataclass(frozen=True)
class BootstrapConfig:
    enabled: bool = True
    n_resamples: int = 5000
    ci: float = 0.95

@dataclass(frozen=True)
class StatsConfig:
    paired: bool = True
    alpha: float = 0.05
    multiple_comparison: Literal["none", "bonferroni", "fdr_bh"] = "fdr_bh"
    tests: Tuple[Literal["ttest_paired", "wilcoxon"], ...] = ("ttest_paired", "wilcoxon")
    effect_sizes: Tuple[Literal["cohens_dz", "cliffs_delta"], ...] = ("cohens_dz", "cliffs_delta")
    bootstrap: BootstrapConfig = BootstrapConfig()
    group_by: Tuple[Literal["seq"], ...] = ("seq",)
    compare_methods: Optional[Tuple[Tuple[str, str], ...]] = None

@dataclass(frozen=True)
class IOConfig:
    output_dir: Path = Path("results/analysis")
    write_csv: bool = True
    write_json: bool = True
    save_plots: bool = False
    overwrite: bool = True

@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 1337
    num_workers: int = 8
    chunk_voxels: int = 20_000_000
    progress: bool = True
    fail_fast: bool = False

@dataclass(frozen=True)
class AnalysisConfig:
    data: AnalysisDataConfig
    metrics: MetricsConfig
    stats: StatsConfig
    io: IOConfig
    runtime: RuntimeConfig

# -----------------------------
# Pydantic validators (strict parsing)
# -----------------------------

class _NormModel(BaseModel):
    kind: Literal["none", "zscore", "minmax"] = "none"
    clip: Optional[Tuple[float, float]] = None

class _SSIMModel(BaseModel):
    win_size: int = 7
    gaussian_weights: bool = True
    sigma: float = 1.5
    use_sample_covariance: bool = False
    data_range: Union[Literal["auto"], float] = "auto"

class _PSNRModel(BaseModel):
    data_range: Union[Literal["auto"], float] = "auto"

class _MetricsModel(BaseModel):
    compute: List[str] = Field(default_factory=lambda: ["psnr", "ssim", "mae", "rmse", "ncc"])
    crop_border: int = 0
    ssim: _SSIMModel = _SSIMModel()
    psnr: _PSNRModel = _PSNRModel()

class _DataModel(BaseModel):
    gt_dir: Path
    pred_dirs: Dict[str, Path]
    mask_dir: Optional[Path] = None
    file_ext: Literal[".nii.gz", ".nii", ".npy", ".npz"] = ".nii.gz"
    sequences: List[str] = Field(default_factory=lambda: ["T1C", "T1N", "T2W", "T2F"])
    cases_list: Optional[Path] = None
    filename_pattern: str = "{case}_{seq}"
    allow_missing: bool = False
    dtype: Literal["float32", "float64"] = "float32"
    norm: _NormModel = _NormModel()

    @field_validator("gt_dir", "mask_dir", mode='before')
    @classmethod
    def _expand_dirs(cls, v):
        return Path(v).expanduser().resolve() if v is not None else None

    @field_validator("pred_dirs", mode='before')
    @classmethod
    def _expand_pred_dirs(cls, v):
        return {k: Path(p).expanduser().resolve() for k, p in v.items()}

class _BootstrapModel(BaseModel):
    enabled: bool = True
    n_resamples: int = 5000
    ci: float = 0.95

class _StatsModel(BaseModel):
    paired: bool = True
    alpha: float = 0.05
    multiple_comparison: Literal["none", "bonferroni", "fdr_bh"] = "fdr_bh"
    tests: List[Literal["ttest_paired", "wilcoxon"]] = Field(default_factory=lambda: ["ttest_paired","wilcoxon"])
    effect_sizes: List[Literal["cohens_dz","cliffs_delta"]] = Field(default_factory=lambda: ["cohens_dz","cliffs_delta"])
    bootstrap: _BootstrapModel = _BootstrapModel()
    group_by: List[Literal["seq"]] = Field(default_factory=lambda: ["seq"])
    compare_methods: Optional[List[Tuple[str, str]]] = None

class _IOModel(BaseModel):
    output_dir: Path = Path("results/analysis")
    write_csv: bool = True
    write_json: bool = True
    save_plots: bool = False
    overwrite: bool = True

    @field_validator("output_dir", mode='before')
    @classmethod
    def _expand_out(cls, v):
        return Path(v).expanduser().resolve()

class _RuntimeModel(BaseModel):
    seed: int = 1337
    num_workers: int = 8
    chunk_voxels: int = 20_000_000
    progress: bool = True
    fail_fast: bool = False

class _AnalysisModel(BaseModel):
    data: _DataModel
    metrics: _MetricsModel = _MetricsModel()
    stats: _StatsModel = _StatsModel()
    io: _IOModel = _IOModel()
    runtime: _RuntimeModel = _RuntimeModel()

# -----------------------------
# Public API
# -----------------------------

def parse_analysis_config(yaml_path: str | Path) -> AnalysisConfig:
    """
    Parse and validate the `analysis:` section into an AnalysisConfig instance.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "analysis" not in cfg:
        raise ValueError("YAML missing top-level 'analysis' key.")
    try:
        model = _AnalysisModel(**cfg["analysis"])
    except ValidationError as e:
        raise ValueError(f"Invalid analysis configuration: {e}") from e

    d = model.dict()
    return AnalysisConfig(
        data=AnalysisDataConfig(**d["data"]),
        metrics=MetricsConfig(
            compute=tuple(d["metrics"]["compute"]),
            crop_border=d["metrics"]["crop_border"],
            ssim=SSIMConfig(**d["metrics"]["ssim"]),
            psnr=PSNRConfig(**d["metrics"]["psnr"]),
        ),
        stats=StatsConfig(
            paired=d["stats"]["paired"],
            alpha=d["stats"]["alpha"],
            multiple_comparison=d["stats"]["multiple_comparison"],
            tests=tuple(d["stats"]["tests"]),
            effect_sizes=tuple(d["stats"]["effect_sizes"]),
            bootstrap=BootstrapConfig(**d["stats"]["bootstrap"]),
            group_by=tuple(d["stats"]["group_by"]),
            compare_methods=tuple(map(tuple, d["stats"]["compare_methods"])) if d["stats"]["compare_methods"] else None,
        ),
        io=IOConfig(**d["io"]),
        runtime=RuntimeConfig(**d["runtime"]),
    )


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data paths and CV manifest generation."""
    models_root: Path
    hr_root: Path
    lr_root: Path
    spacings: Tuple[str, ...]
    pulses: Tuple[str, ...]
    models: Tuple[str, ...]
    kfolds: int
    seed: int
    out: Path


@dataclass(frozen=True)
class ExpertConfig:
    """Configuration for a single expert model in prediction."""
    name: str
    opinion_path: Path


@dataclass(frozen=True)
class PredictConfig:
    """Configuration for prediction mode."""
    pacs_sr_weights_path: Path
    experts: Dict[str, ExpertConfig]
    out_root: Path


@dataclass(frozen=True)
class PacsSRConfig:
    # identity
    experiment_name: str
    # tiling
    patch_size: int
    stride: int
    # optimization
    simplex: bool
    lambda_ridge: float
    laplacian_tau: float
    # edge weighting
    lambda_edge: float
    edge_power: float
    # gradient-domain augmentation
    lambda_grad: float
    grad_operator: str
    # overlap-add blending
    mixing_window: str
    # normalization
    normalize: str
    # registration
    use_registration: bool
    atlas_dir: Optional[Path]
    # compute
    num_workers: int
    parallel_backend: str
    device: str
    # metrics
    compute_lpips: bool
    compute_kid: bool
    ssim_axis: str
    evaluate_train: bool
    # saving
    save_weight_volumes: bool
    save_blends: bool
    # logging
    log_level: str
    log_to_file: bool
    log_region_freq: int
    disable_tqdm: bool  # Disable tqdm progress bars (use SLURM-friendly logging instead)
    # paths
    out_root: Path
    # data fields (populated from DataConfig in CLI)
    models: Tuple[str, ...]
    spacings: Tuple[str, ...]
    pulses: Tuple[str, ...]


@dataclass(frozen=True)
class RegistrationConfig:
    """Configuration for atlas-based registration."""
    atlas: Path
    t1: str
    t2: str
    epi: str
    pd: str
    brain_mask: str


@dataclass(frozen=True)
class FullConfig:
    """Complete configuration containing all sections."""
    data: DataConfig
    registration: RegistrationConfig
    pacs_sr: PacsSRConfig
    predict: Optional[PredictConfig] = None
    # Optional visualization block
    visualizations: Optional["VisualizationsConfig"] = None
    # Optional pipeline orchestration block
    pipeline: Optional["PipelineConfig"] = None


@dataclass(frozen=True)
class VisualizationsConfig:
    """Configuration for visualization outputs and inputs."""
    results_root: Path   # root where PaCS-SR wrote results (experiment directory lives here)
    out_root: Path       # output directory for figures/reports
    # Future extensions: figure dpi, slice selection policy, etc.
    dpi: int = 200


@dataclass(frozen=True)
class FiguresConfig:
    """Configuration for publication figure generation."""
    dpi: int = 300
    format: Tuple[str, ...] = ("pdf", "png")
    generate_metrics_table: bool = True
    generate_boxplots: bool = True
    generate_weight_heatmaps: bool = True
    generate_patient_examples: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the end-to-end pipeline orchestrator."""
    experiment_name: str
    output_root: Path
    timestamp_suffix: bool = True
    resume: bool = True
    max_retries: int = 2
    # Stages to run
    run_setup: bool = True
    run_manifest: bool = True
    run_training: bool = True
    run_analysis: bool = True
    run_visualization: bool = True
    run_report: bool = False
    # Figure settings
    figures: FiguresConfig = field(default_factory=FiguresConfig)

def load_data_config(config_dict: dict) -> DataConfig:
    """Parse the data section of the config."""
    return DataConfig(
        models_root=Path(config_dict["models-root"]),
        hr_root=Path(config_dict["hr-root"]),
        lr_root=Path(config_dict["lr-root"]),
        spacings=tuple(config_dict["spacings"]),
        pulses=tuple(config_dict["pulses"]),
        models=tuple(config_dict["models"]),
        kfolds=int(config_dict["kfolds"]),
        seed=int(config_dict["seed"]),
        out=Path(config_dict["out"])
    )


def load_registration_config(config_dict: dict) -> RegistrationConfig:
    """Parse the registration section of the config."""
    return RegistrationConfig(
        atlas=Path(config_dict["atlas"]),
        t1=str(config_dict["t1"]),
        t2=str(config_dict["t2"]),
        epi=str(config_dict["epi"]),
        pd=str(config_dict["pd"]),
        brain_mask=str(config_dict["brain_mask"])
    )


def load_pacs_sr_config(
    config_dict: dict,
    data_config: Optional[DataConfig] = None,
    registration_config: Optional[RegistrationConfig] = None
) -> PacsSRConfig:
    """
    Parse the pacs_sr section of the config.

    Args:
        config_dict: The pacs_sr section from YAML
        data_config: Optional DataConfig to populate models/spacings/pulses fields
        registration_config: Optional RegistrationConfig to get atlas_dir

    Returns:
        PacsSRConfig with all fields populated
    """
    # Get models, spacings, pulses from data_config if provided
    if data_config is not None:
        models = data_config.models
        spacings = data_config.spacings
        pulses = data_config.pulses
    else:
        # For backward compatibility, try to get from config_dict
        models = tuple(config_dict.get("models", []))
        spacings = tuple(config_dict.get("spacings", []))
        pulses = tuple(config_dict.get("pulses", []))

    # Get atlas_dir from registration_config if use_registration is enabled
    atlas_dir = None
    if config_dict.get("use_registration", False):
        if registration_config is not None:
            atlas_dir = registration_config.atlas
        elif "atlas_dir" in config_dict and config_dict["atlas_dir"]:
            # Fallback to old config format for backward compatibility
            atlas_dir = Path(config_dict["atlas_dir"])

    return PacsSRConfig(
        experiment_name=config_dict["experiment_name"],
        patch_size=int(config_dict["patch_size"]),
        stride=int(config_dict["stride"]),
        simplex=bool(config_dict["simplex"]),
        lambda_ridge=float(config_dict["lambda_ridge"]),
        laplacian_tau=float(config_dict["laplacian_tau"]),
        lambda_edge=float(config_dict["lambda_edge"]),
        edge_power=float(config_dict["edge_power"]),
        lambda_grad=float(config_dict.get("lambda_grad", 0.0)),
        grad_operator=str(config_dict.get("grad_operator", "sobel")).lower(),
        mixing_window=str(config_dict.get("mixing_window", "flat")).lower(),
        normalize=str(config_dict["normalize"]),
        use_registration=bool(config_dict.get("use_registration", False)),
        atlas_dir=atlas_dir,
        num_workers=int(config_dict["num_workers"]),
        parallel_backend=str(config_dict.get("parallel_backend", "loky")),
        device=str(config_dict["device"]),
        compute_lpips=bool(config_dict.get("compute_lpips", False)),
        compute_kid=bool(config_dict.get("compute_kid", False)),
        ssim_axis=str(config_dict["ssim_axis"]),
        evaluate_train=bool(config_dict.get("evaluate_train", True)),
        save_weight_volumes=bool(config_dict["save_weight_volumes"]),
        save_blends=bool(config_dict["save_blends"]),
        log_level=str(config_dict.get("log_level", "INFO")),
        log_to_file=bool(config_dict.get("log_to_file", True)),
        log_region_freq=int(config_dict.get("log_region_freq", 10)),
        disable_tqdm=bool(config_dict.get("disable_tqdm", False)),
        out_root=Path(config_dict["out_root"]),
        models=models,
        spacings=spacings,
        pulses=pulses
    )


def load_predict_config(config_dict: dict) -> PredictConfig:
    """Parse the predict section of the config."""
    experts = {}
    for expert_id, expert_data in config_dict["experts"].items():
        experts[expert_id] = ExpertConfig(
            name=expert_data["name"],
            opinion_path=Path(expert_data["opinion_path"])
        )

    return PredictConfig(
        pacs_sr_weights_path=Path(config_dict["pacs_sr_weights_path"]),
        experts=experts,
        out_root=Path(config_dict["out_root"])
    )


def load_full_config(path: Path) -> FullConfig:
    """
    Load the complete configuration file with all sections.

    Args:
        path: Path to YAML configuration file

    Returns:
        FullConfig object with data, registration, pacs_sr, and optional predict sections
    """
    with open(path, "r") as f:
        yaml_data = yaml.safe_load(f)

    data_config = load_data_config(yaml_data["data"])
    registration_config = load_registration_config(yaml_data["registration"])
    # Pass data_config and registration_config to populate all fields in pacs_sr_config
    pacs_sr_config = load_pacs_sr_config(
        yaml_data["pacs_sr"],
        data_config=data_config,
        registration_config=registration_config
    )

    predict_config = None
    if "predict" in yaml_data:
        predict_config = load_predict_config(yaml_data["predict"])

    # Optional visualizations section
    viz_config = None
    if "visualizations" in yaml_data:
        v = yaml_data["visualizations"]
        viz_config = VisualizationsConfig(
            results_root=Path(v["results_root"]),
            out_root=Path(v["out_root"]),
            dpi=int(v.get("dpi", 200))
        )

    # Optional pipeline section
    pipeline_config = None
    if "pipeline" in yaml_data:
        p = yaml_data["pipeline"]
        figures_cfg = FiguresConfig()
        if "figures" in p:
            f = p["figures"]
            figures_cfg = FiguresConfig(
                dpi=int(f.get("dpi", 300)),
                format=tuple(f.get("format", ["pdf", "png"])),
                generate_metrics_table=bool(f.get("generate_metrics_table", True)),
                generate_boxplots=bool(f.get("generate_boxplots", True)),
                generate_weight_heatmaps=bool(f.get("generate_weight_heatmaps", True)),
                generate_patient_examples=bool(f.get("generate_patient_examples", True)),
            )
        pipeline_config = PipelineConfig(
            experiment_name=str(p.get("experiment_name", pacs_sr_config.experiment_name)),
            output_root=Path(p.get("output_root", pacs_sr_config.out_root)),
            timestamp_suffix=bool(p.get("timestamp_suffix", True)),
            resume=bool(p.get("resume", True)),
            max_retries=int(p.get("max_retries", 2)),
            run_setup=bool(p.get("stages", {}).get("setup", True)),
            run_manifest=bool(p.get("stages", {}).get("manifest", True)),
            run_training=bool(p.get("stages", {}).get("training", True)),
            run_analysis=bool(p.get("stages", {}).get("analysis", True)),
            run_visualization=bool(p.get("stages", {}).get("visualization", True)),
            run_report=bool(p.get("stages", {}).get("report", False)),
            figures=figures_cfg,
        )

    return FullConfig(
        data=data_config,
        registration=registration_config,
        pacs_sr=pacs_sr_config,
        predict=predict_config,
        visualizations=viz_config,
        pipeline=pipeline_config,
    )


def load_config(path: Path) -> PacsSRConfig:
    """
    Parse the YAML experiment configuration file into a strongly typed dataclass.

    DEPRECATED: Use load_full_config() for new code.
    This function is kept for backward compatibility.
    """
    full_config = load_full_config(path)
    return full_config.pacs_sr
