from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self, _input_dim: int, _hidden_dim: int, _output_dim: int
    ) -> None:
        super().__init__()

        self.fc_1 = nn.Linear(_input_dim, _hidden_dim)
        self.fc_2 = nn.Linear(_hidden_dim, _output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)

        return x


def init_weights(m: Union[MLP, nn.Linear]) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
