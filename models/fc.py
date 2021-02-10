import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init, SELU, Sequential, Linear, ReLU
import math


def fc1(num_classes):
    return torch.nn.Sequential(
        Linear(96, 256),
        SELU(),
        Linear(256, 256),
        SELU(),
        Linear(256, 256),
        SELU(),
        Linear(256, num_classes),
    )

# def fc1(num_classes):
#     return torch.nn.Sequential(
#         Linear(2, 256),
#         SELU(),
#         Linear(256, 256),
#         SELU(),
#         # Linear(256, 256),
#         # SELU(),
#         Linear(256, num_classes),
#     )