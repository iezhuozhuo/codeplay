# -*- coding: utf-8 -*-
# @Time    : 2020/7/17 15:05
# @Author  : zhuo & zdy
# @github   : iezhuozhuo
import typing

import torch
from torch import nn


activation = nn.ModuleDict([
    ['relu', nn.ReLU()],
    ['hardtanh', nn.Hardtanh()],
    ['relu6', nn.ReLU6()],
    ['sigmoid', nn.Sigmoid()],
    ['tanh', nn.Tanh()],
    ['softmax', nn.Softmax()],
    ['softmax2d', nn.Softmax2d()],
    ['logsoftmax', nn.LogSoftmax()],
    ['elu', nn.ELU()],
    ['selu', nn.SELU()],
    ['celu', nn.CELU()],
    ['hardshrink', nn.Hardshrink()],
    ['leakyrelu', nn.LeakyReLU()],
    ['logsigmoid', nn.LogSigmoid()],
    ['softplus', nn.Softplus()],
    ['softshrink', nn.Softshrink()],
    ['prelu', nn.PReLU()],
    ['softsign', nn.Softsign()],
    ['softmin', nn.Softmin()],
    ['tanhshrink', nn.Tanhshrink()],
    ['rrelu', nn.RReLU()],
    ['glu', nn.GLU()],
])


def _parse(
    identifier: typing.Union[str, typing.Type[nn.Module], nn.Module],
    dictionary: nn.ModuleDict,
    target: str
) -> nn.Module:
    """
    Parse loss and activation.
    :param identifier: activation identifier, one of
            - String: name of a activation
            - Torch Modele subclass
            - Torch Module instance (it will be returned unchanged).
    :param dictionary: nn.ModuleDict instance. Map string identifier to
        nn.Module instance.
    :return: A :class:`nn.Module` instance
    """
    if isinstance(identifier, str):
        if identifier in dictionary:
            return dictionary[identifier]
        else:
            raise ValueError(
                f'Could not interpret {target} identifier: ' + str(identifier)
            )
    elif isinstance(identifier, nn.Module):
        return identifier
    elif issubclass(identifier, nn.Module):
        return identifier()
    else:
        raise ValueError(
            f'Could not interpret {target} identifier: ' + str(identifier)
        )


def parse_activation(
    identifier: typing.Union[str, typing.Type[nn.Module], nn.Module]
) -> nn.Module:
    """
    Retrieves a torch Module instance.
    :param identifier: activation identifier, one of
            - String: name of a activation
            - Torch Modele subclass
            - Torch Module instance (it will be returned unchanged).
    :return: A :class:`nn.Module` instance
    Examples::
        >>> from torch import nn
        >>> from source.modules.activate import parse_activation
    Use `str` as activation:
        >>> activation = parse_activation('relu')
        >>> type(activation)
        <class 'torch.nn.modules.activation.ReLU'>
    Use :class:`torch.nn.Module` subclasses as activation:
        >>> type(parse_activation(nn.ReLU))
        <class 'torch.nn.modules.activation.ReLU'>
    Use :class:`torch.nn.Module` instances as activation:
        >>> type(parse_activation(nn.ReLU()))
        <class 'torch.nn.modules.activation.ReLU'>
    """

    return _parse(identifier, activation, 'activation')
