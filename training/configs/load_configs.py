import os
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import yaml

_NUMERIC_TOPLEVEL = {
    "batchsize",
    "lr",
    "trainsize",
    "decay_rate",
    "clip",
    "epoch",
    "decay_epoch",
    "scheduler_patience",
}
_PATH_KEYS = ("train_path", "test_path", "train_save")


def _is_uri(path_value: str) -> bool:
    """Return True when the path string looks like a remote URI."""
    try:
        scheme = urlparse(str(path_value)).scheme
    except ValueError:
        return False
    return bool(scheme) and scheme != "file"


def _to_number(val: Any) -> Any:
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, list):
        return [_to_number(v) for v in val]
    if isinstance(val, str):
        s = val.strip()
        try:
            if s.isdigit():
                return int(s)
            return float(s)
        except ValueError:
            return os.path.expandvars(s)
    return val


def _normalize_paths(cfg: Dict[str, Any]) -> None:
    for key in _PATH_KEYS:
        value = cfg.get(key)
        if not value:
            continue
        if _is_uri(value):
            cfg[key] = str(value)
        else:
            cfg[key] = str(Path(value).expanduser().resolve())


def _normalize_numbers(cfg: Dict[str, Any]) -> None:
    for key in list(cfg.keys()):
        if key in _NUMERIC_TOPLEVEL:
            cfg[key] = _to_number(cfg[key])
    tuning = cfg.get("tuning")
    if isinstance(tuning, dict):
        for key, value in list(tuning.items()):
            if key == "search_space" and isinstance(value, dict):
                tuning[key] = {
                    sub_key: _to_number(sub_val) for sub_key, sub_val in value.items()
                }
            else:
                tuning[key] = _to_number(value)


def _validate(cfg: Dict[str, Any]) -> None:
    """Validate essential config fields and paths."""

    validate_paths = bool(cfg.get("validate_paths", True))
    if not validate_paths:
        cfg.pop("validate_paths", None)
    else:
        for k in ("train_path", "test_path"):
            p = cfg.get(k)
            if p and not Path(p).exists():
                raise FileNotFoundError(f"{k} not found: {p}")

    save_dir = Path(cfg.get("train_save", "./model_pth")).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg["train_save"] = str(save_dir.resolve())


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from a YAML file."""

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        cfg: Dict[str, Any] = yaml.safe_load(config_file.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {exc}") from exc

    def _expand(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    cfg = _expand(cfg)
    _normalize_paths(cfg)
    _normalize_numbers(cfg)
    _validate(cfg)

    return cfg
