from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

CONFIG_DIR = Path(Path(__file__).parents[1], "config")


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
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


training_config = load_config("training")


def plot_reward(
    _train_rewards: List,
    _test_rewards: List,
    _reward_thres: int = None,
    save: Optional[str] = None,
) -> None:
    """
    Plot the rewards of RL trajectories.

    Parameters
    ----------
    _train_rewards : List
        Train rewards to plot
    _test_rewards : List
        Test rewards to plot
    _reward_thres : int
        Reward threshold (indicate with red line)
    save: str
        If specified, save plot with name in plots directory

    Returns
    -------
    None
    """
    plt.figure(figsize=training_config["plot_figsize"])
    plt.plot(_test_rewards, label="Test Reward")
    plt.plot(_train_rewards, label="Train Reward")
    plt.xlabel("Episode", fontsize=training_config["axis_fontsize"])
    plt.ylabel("Reward", fontsize=training_config["axis_fontsize"])
    if _reward_thres:
        plt.hlines(_reward_thres, 0, len(_test_rewards), color="r")

    plt.legend(loc="lower right")
    plt.grid()

    if save is not None:
        plt.savefig(Path(training_config["plots_dir"], save))

    plt.show()


def seed_everything(seed: int = training_config["seed"]) -> None:
    """
    Seed numpy and PyTorch.

    Parameters
    ----------
    seed : int
        Seed to set.

    Returns
    -------
    None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
