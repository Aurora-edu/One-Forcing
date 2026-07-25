from pathlib import Path
from typing import Set

from omegaconf import DictConfig, OmegaConf


def _load_config(path: Path, active: Set[Path]) -> DictConfig:
    path = path.expanduser().resolve()
    if path in active:
        chain = " -> ".join(str(item) for item in [*active, path])
        raise ValueError(f"Cyclic _base_ config inheritance: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    active.add(path)
    config = OmegaConf.load(path)
    base_value = config.pop("_base_", None)
    if base_value is None:
        merged = config
    else:
        base_paths = [base_value] if isinstance(base_value, str) else list(base_value)
        bases = []
        for base_path in base_paths:
            resolved = Path(base_path)
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            bases.append(_load_config(resolved, active))
        merged = OmegaConf.merge(*bases, config)
    active.remove(path)
    return merged


def load_config(path: str) -> DictConfig:
    """Load an OmegaConf YAML file with optional relative ``_base_`` inheritance."""
    return _load_config(Path(path), set())
