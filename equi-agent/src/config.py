from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON/YAML config file as a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML configs.") from exc
        loaded = yaml.safe_load(text)
        return loaded or {}
    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a config dictionary as JSON or YAML based on file extension."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        return
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to save YAML configs.") from exc
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return
    raise ValueError(f"Unsupported config format: {config_path.suffix}")

