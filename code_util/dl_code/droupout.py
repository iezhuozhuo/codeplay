# 这个是keras上的实现dropout
# level是概率p屏蔽激活值
# def dropout(x, level, noise_shape=None, seed=None):
#     """Sets entries in `x` to zero at random,
#     while scaling the entire tensor.
#     # Arguments
#         x: tensor
#         level: fraction of the entries in the tensor
#             that will be set to 0.
#         noise_shape: shape for randomly generated keep/drop flags,
#             must be broadcastable to the shape of `x`
#         seed: random seed to ensure determinism.
#     """
#     if level < 0. or level >= 1:
#         raise ValueError('Dropout level must be in interval [0, 1[.')
#     if seed is None:
#         seed = np.random.randint(1, 10e6)
#     if isinstance(noise_shape, list):
#         noise_shape = tuple(noise_shape)
#
#     rng = RandomStreams(seed=seed)
#     retain_prob = 1. - level
#
#     if noise_shape is None:
#         random_tensor = rng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
#     else:
#         random_tensor = rng.binomial(noise_shape, p=retain_prob, dtype=x.dtype)
#         random_tensor = T.patternbroadcast(random_tensor,
#                                            [dim == 1 for dim in noise_shape])
#     x *= random_tensor
#     x /= retain_prob
#     return x

# 简单实现dropout
import numpy as np


def dropout(x, level):
    if level < 0. or level >= 1: #level是概率值，必须在0~1之间
        raise ValueError('Dropout level must be in interval [0, 1[.')
    retain_prob = 1. - level
    # 我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
    # 硬币 正面的概率为p，n表示每个神经元试验的次数
    # 因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
    random_tensor = np.random.binomial(n=1, p=retain_prob, size=x.shape) #即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
    print(random_tensor)

    x *= random_tensor
    print(x)
    x /= retain_prob

    return x


x = np.asarray([3, 14, 15, 9, 2, 6], dtype=np.float32)
print(x.shape)
print(dropout(x, 0.4))
