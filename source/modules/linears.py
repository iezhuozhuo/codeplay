import torch
import torch.nn as nn
import torch.nn.functional as F

from source.modules.activate import parse_activation


# TODO 重复定义
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


def quickly_perceptron_layer(
        in_features: int = 0,
        out_features: int = 0,
        activation: nn.Module = nn.ReLU()
) -> nn.Module:
    """:return: a perceptron layer."""
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        activation
    )


def quickly_multi_layer_perceptron_layer(
        in_features,
        mlp_num_layers=1,
        mlp_num_units=128,
        mlp_num_fan_out=64,
        mlp_activation_func='relu') -> nn.Module:
    """:return: a multiple layer perceptron."""

    activation = parse_activation(mlp_activation_func)
    mlp_sizes = [
        in_features,
        *mlp_num_layers * [mlp_num_units],
        mlp_num_fan_out
    ]
    mlp = [
        quickly_perceptron_layer(in_f, out_f, activation)
        for in_f, out_f in zip(mlp_sizes, mlp_sizes[1:])
    ]
    return nn.Sequential(*mlp)


def quickly_output_layer(
        task,
        num_classes: int = 1,
        in_features: int = 0,
        out_activation_func: str = ''
) -> nn.Module:
    """:return: a correctly shaped torch module for model output."""
    if task == "classify":
        out_features = num_classes
    elif task == "Ranking":
        out_features = 1
    else:
        raise ValueError(f"{task} is not a valid task type. "
                         f"Must be in `Ranking` and `Classification`.")
    if out_activation_func:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            parse_activation(out_activation_func)
        )
    else:
        return nn.Linear(in_features, out_features)
