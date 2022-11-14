import torch.nn as nn


def init_weights_xav(m: nn.Module) -> None:
    """
    Initialize Xavier normal weights for a layer.

    Parameters
    ----------
    m : nn.Module
        Layer whose weights to initialize.

    Returns
    -------
    None
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
