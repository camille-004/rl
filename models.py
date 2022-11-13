from typing import Optional, Sequence, Type, Union

import torch
import torch.nn as nn


class MLP(nn.Module):
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

        print(activ_layers)
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
        return self.model(x)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
