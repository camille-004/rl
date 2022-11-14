from abc import ABC
from typing import List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import Agent

from src.models.models import MLP
from src.models.utils import init_weights_xav
from src.utils import load_config, plot_reward, seed_everything

training_config = load_config("training")
vpg_config = load_config("vpg")

DISCOUNT_FACTOR = vpg_config["discount_factor"]
HIDDEN_DIM = vpg_config["hidden_dim"]
LEARNING_RATE = vpg_config["lr"]
REWARD_THRES = vpg_config["reward_thres"]

seed_everything()
SEED = training_config["seed"]

MAX_TRAJECTORY = training_config["max_trajectory"]
N_TRIALS = training_config["n_trials"]
PRINT_EVERY = training_config["print_every"]


class VanillaPolicyGradient(Agent, ABC):
    """Class to train and run a vanilla policy gradient with an MLP."""

    def __init__(
        self,
        _train_env: gym.Env,
        _test_env: Optional[gym.Env] = None,
        dist_fn: Type[
            torch.distributions.Distribution
        ] = torch.distributions.Categorical,
        discount_factor: float = DISCOUNT_FACTOR,
        lr: float = LEARNING_RATE,
        hidden_dim: Union[int, Sequence[int]] = HIDDEN_DIM,
        norm_layers: Optional[
            Union[Type[nn.Module], Sequence[Type[nn.Module]]]
        ] = None,
        activ_layers: Optional[
            Union[Type[nn.Module], Sequence[Type[nn.Module]]]
        ] = nn.ReLU,
        normalize_returns: bool = True,
        render: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        _train_env : gym.Env
            Environment to train on.
        _test_env : Optional[gym.Env]
            Separate environment for evaluation. Optional.
        dist_fn : Type[torch.distributions.Distribution]
            PyTorch distribution from which to sample actions.
        discount_factor : float
            Discount factor.
        lr : float
            Learning rate for Adam optimizer.
        hidden_dim : Union[int, Sequence[int]]
            Hidden dimensions for MLP. Can pass a single hidden size for a
            single-layer perceptron, or a list of dimensions.
        norm_layers : Optional[
            Union[Type[nn.Module], Sequence[Type[nn.Module]]]
        ]
            Normalization layers (i.e., nn.LayerNorm). Can pass in a single
            type to use across all layers or a list of layers. Must be the
            same size as the hidden_dim list, if passed as a list.
        activ_layers : Optional[
            Union[Type[nn.Module], Sequence[Type[nn.Module]]]
        ]
            Activation layers (i.e., nn.Tanh). Can pass in a single
            type to use across all layers or a list of layers. Must be the
            same size as the hidden_dim list, if passed as a list.
        normalize_returns : bool
            Whether to normalize returns.
        render : bool
            Whether to render the environment during training.
        """
        super().__init__(_train_env, _test_env, render)

        assert (
            0.0 <= discount_factor <= 1.0
        ), "Discount factor must be in [0, 1]"
        self.train_env = _train_env

        if _test_env:
            self.test_env = _test_env

        self.input_dim = self.train_env.observation_space.shape[0]

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        self.hidden_dim = hidden_dim
        self.output_dim = self.train_env.action_space.n
        norm_layers = norm_layers
        activ_layers = activ_layers

        self.discount_factor = discount_factor

        self.policy = MLP(
            self.input_dim,
            self.output_dim,
            self.hidden_dim,
            norm_layers,
            activ_layers,
        )
        self.policy.apply(init_weights_xav)
        print(self.policy)

        self.dist_fn = dist_fn
        self.lr = lr
        self.normalize_returns = normalize_returns

        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def act(self, state_obs: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Choose an action from a forward pass through the MLP. Return the log
        probability of the selected action and the action itself.

        Parameters
        ----------
        state_obs : torch.Tensor
            Current observation.

        Returns
        -------
        Tuple[torch.Tensor, int]
            The log probability of the selected action and the action itself.
        """
        action_pred = self.policy(state_obs)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = self.dist_fn(action_prob)
        action = dist.sample()
        action_lob_prob = dist.log_prob(action)
        return action_lob_prob, action.item()

    def update_policy(
        self, returns: List, action_log_probs: torch.Tensor
    ) -> float:
        """
        Backward pass through the MLP. Maximize the objective function.

        Parameters
        ----------
        returns : List
            Rewards for the objective function.
        action_log_probs : torch.Tensor
            Log likelihood for the objective function.

        Returns
        -------
        float
            Loss after backward pass.
        """
        # No gradient for returns, detach from computational graph
        # returns = returns.detach()

        # Objective function, J(\pi_\theta)
        _loss = -(returns * action_log_probs).sum()

        self.opt.zero_grad()
        _loss.backward()
        self.opt.step()
        return _loss.item()

    def calc_returns(self, rewards: List) -> List:
        """
        Calculate the discounted returns.

        Parameters
        ----------
        rewards : List
            Input list of rewards.

        Returns
        -------
        List
            Discounted returns.
        """
        returns = []
        _R = 0

        for r in reversed(rewards):
            _R = r + _R * self.discount_factor
            returns.insert(0, _R)

        returns = torch.Tensor(returns)

        if self.normalize_returns:
            returns = (returns - returns.mean()) / returns.std(dim=-1)

        return returns

    def train(self) -> Tuple[float, float]:
        """
        Train an episode of policy. Return the loss after updating the policy
        and the reward for the current episode.

        Returns
        -------
        Tuple[float, float]
            The loss and reward for the current episode.
        """
        self.policy.train()

        done = False
        action_log_probs = []
        rewards = []
        ep_reward = 0

        state = self.train_env.reset()

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)

            if self.render:
                self.train_env.render()

            action_log_prob, action = self.act(state)
            state, reward, done, _ = self.train_env.step(action)
            action_log_probs.append(action_log_prob)
            rewards.append(reward)
            ep_reward += reward

        action_log_probs = torch.cat(action_log_probs)
        returns = self.calc_returns(rewards)
        _loss = self.update_policy(returns, action_log_probs)

        return _loss, ep_reward

    def evaluate(self) -> float:
        """
        Evaluate an episode of the policy. Evaluate on the test_env, if
        provided.

        Returns
        -------
        float
            Test reward for the current episode.
        """
        self.policy.eval()

        done = False
        ep_reward = 0

        if self.test_env:
            eval_env = self.test_env
        else:
            eval_env = self.train_env

        state = eval_env.reset()

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action_pred = self.policy(state)
                action_prob = F.softmax(action_pred, dim=-1)

            action = torch.argmax(action_prob, dim=-1)
            state, reward, done, _ = eval_env.step(action.item())
            ep_reward += reward

        return ep_reward


if __name__ == "__main__":
    train_env = gym.make("CartPole-v1")
    test_env = gym.make("CartPole-v1")
    train_env.seed(SEED)
    test_env.seed(SEED + 1)
    vpg = VanillaPolicyGradient(train_env, test_env, activ_layers=nn.Tanh)

    train_rewards = []
    test_rewards = []

    for ep in range(1, MAX_TRAJECTORY + 1):
        loss, train_reward = vpg.train()
        test_reward = vpg.evaluate()

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if ep % PRINT_EVERY == 0:
            print(
                f"| Episode: {ep:3} | Mean Train Rewards: "
                f"{mean_train_rewards:5.1f} | Mean Test Rewards: "
                f"{mean_test_rewards:5.1f}"
            )
        if mean_test_rewards >= REWARD_THRES:
            print(f"Reached reward threshold in {ep} episodes")
            break

    vpg.close_env()
    plot_reward(
        train_rewards, test_rewards, REWARD_THRES, save=vpg_config["plot_name"]
    )
