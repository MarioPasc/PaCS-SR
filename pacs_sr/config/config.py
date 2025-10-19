from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml


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
    device: str
    # metrics
    compute_lpips: bool
    ssim_axis: str
    evaluate_train: bool
    # saving
    save_weight_volumes: bool
    save_blends: bool
    # logging
    log_level: str
    log_to_file: bool
    log_region_freq: int
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
        device=str(config_dict["device"]),
        compute_lpips=bool(config_dict["compute_lpips"]),
        ssim_axis=str(config_dict["ssim_axis"]),
        evaluate_train=bool(config_dict.get("evaluate_train", True)),
        save_weight_volumes=bool(config_dict["save_weight_volumes"]),
        save_blends=bool(config_dict["save_blends"]),
        log_level=str(config_dict.get("log_level", "INFO")),
        log_to_file=bool(config_dict.get("log_to_file", True)),
        log_region_freq=int(config_dict.get("log_region_freq", 10)),
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

    return FullConfig(
        data=data_config,
        registration=registration_config,
        pacs_sr=pacs_sr_config,
        predict=predict_config
    )


def load_config(path: Path) -> PacsSRConfig:
    """
    Parse the YAML experiment configuration file into a strongly typed dataclass.

    DEPRECATED: Use load_full_config() for new code.
    This function is kept for backward compatibility.
    """
    full_config = load_full_config(path)
    return full_config.pacs_sr
