from typing import Any, Dict
import yaml
import pathlib

def load_config(path: str = "conf/project.yaml") -> Dict[str, Any]:
    return yaml.safe_load(pathlib.Path(path).read_text())
