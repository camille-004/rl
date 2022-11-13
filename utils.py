from pathlib import Path
from typing import Dict

import yaml

CONFIG_DIR = "config"


def load_config(config_name: str) -> Dict:
    """
    Load contents from a config file.

    Parameters
    ----------
    config_name : str
        Name of config.

    Returns
    -------
    Dict
        Contents of config file.
    """
    with open(Path(CONFIG_DIR, config_name + ".yaml")) as f:
        config = yaml.safe_load(f)

    return config
