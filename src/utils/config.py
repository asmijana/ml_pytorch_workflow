from dataclasses import dataclass
from pathlib import Path
import yaml

def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f: 
        return yaml.safe_load(f)
    
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p