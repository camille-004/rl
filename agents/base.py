from typing import Optional

import gym


class Agent:
    def __init__(
        self,
        _train_env: gym.Env,
        _test_env: Optional[gym.Env] = None,
        render: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        _train_env : gym.Env
            Environment to train on.
        _test_env : Optional[gym.Env]
            Separate environment for evaluation. Optional.
        render : bool
            Whether to render the environment during training.
        """
        self.train_env = _train_env

        if _test_env:
            self.test_env = _test_env

        self.render = render

    def close_env(self) -> None:
        """
        Close the environment if rendering during training.

        Returns
        -------
        None
        """
        if self.render:
            if self.test_env:
                self.test_env.close()
            else:
                self.train_env.close()
        else:
            return None

    def train(self) -> None:
        """
        Base train method.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def evaluate(self) -> None:
        """
        Base evaluate method.

        Returns
        -------
        None
        """
        raise NotImplementedError
