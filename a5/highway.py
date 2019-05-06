#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn


class Highway(nn.Module):
    def __init__(self, features):
        super(Highway, self).__init__()
        self.proj = nn.Linear(in_features=features, out_features=features)
        self.gate = nn.Linear(in_features=features, out_features=features)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        x_proj = torch.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.proj(x_conv_out))
        return x_gate * x_proj + (1 - x_gate) * x_conv_out
### END YOUR CODE 

