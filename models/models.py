from typing import Optional, Sequence, Type, Union

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Class to build an MLP."""

    def __init__(
        self,
        _input_dim: int,
        _output_dim: int,
        _hidden_dims: Sequence[int] = (),
        norm_layers: Optional[
            Union[Type[nn.Module], Sequence[Type[nn.Module]]]
        ] = None,
        activ_layers: Optional[
            Union[Type[nn.Module], Sequence[Type[nn.Module]]]
        ] = nn.ReLU,
    ) -> None:
        """
        Parameters
        ----------
        _input_dim : int
            Input dimensions for MLP (size of observation space).
        _output_dim : int
            Output dimensions for MLP (size of action space).
        _hidden_dims : Sequence[int]
            List of hidden dimensions for MLP.
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
        """
        super().__init__()

        if norm_layers:
            if isinstance(norm_layers, list):
                assert len(norm_layers) == len(_hidden_dims)
            else:
                norm_layers = [norm_layers] * len(_hidden_dims)
        else:
            norm_layers = [None] * len(_hidden_dims)

        if activ_layers:
            if isinstance(activ_layers, list):
                assert len(activ_layers) == len(_hidden_dims)
            else:
                activ_layers = [activ_layers] * len(_hidden_dims)
        else:
            activ_layers = [None] * len(_hidden_dims)

        dims = [_input_dim] + list(_hidden_dims)
        model = []

        for in_dim, out_dim, norm, activ in zip(
            dims[:-1], dims[1:], norm_layers, activ_layers
        ):
            layer = [nn.Linear(in_dim, out_dim)]
            if norm is not None:
                layer.append(norm(out_dim))
            if activ is not None:
                layer.append(activ())

            model += layer

        model.append(nn.Linear(dims[-1], _output_dim))
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Feature representation of x.
        """
        return self.model(x)
