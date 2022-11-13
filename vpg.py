from typing import Type

import gym
import torch

from utils import load_config

vpg_config = load_config("vpg")


class VanillaPolicyGradient:
    def __init__(
        self,
        env: gym.Env,
        dist_fn: Type[
            torch.distributions.Distribution
        ] = torch.distributions.Categorical,
        discount_factor: float = vpg_config["discount_factor"],
        lr: float = vpg_config["lr"],
        hidden_dim: int = vpg_config["hidden_dim"],
    ):
        assert (
            0.0 <= discount_factor <= 1.0
        ), "Discount factor must be in [0, 1]"
        self.env = env
        self.input_dim = self.env.observation_space.shape[0]
        self.hidden_dim = hidden_dim
        self.output_dim = self.env.action_space.n

        self.dist_fn = dist_fn
        self.lr = lr
        self.opt = torch.optim.Adam(self.policy.paramaters(), lr=self.lr)
