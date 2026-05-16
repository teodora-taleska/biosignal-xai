import yaml
from pathlib import Path

# This file is at src/utils/config.py
# So project root is 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config(path: str = "configs/config.yaml") -> dict:
    config_path = PROJECT_ROOT / path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve all paths to absolute, works from anywhere
    # Trailing '/' preserved so callers can concatenate filenames directly
    if 'preprocessing' in cfg:
        cfg['preprocessing']['path'] = str(PROJECT_ROOT / cfg['preprocessing']['path']) + '/'
    for key in cfg['paths']:
        cfg['paths'][key] = str(PROJECT_ROOT / cfg['paths'][key]) + '/'

    return cfg

CFG = load_config()
