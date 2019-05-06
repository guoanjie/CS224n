#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, channels, kernel_size):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        self.maxpool1d = nn.MaxPool1d(kernel_size=21 - kernel_size + 1)

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv1d(x_reshaped)
        return self.maxpool1d(torch.relu(x_conv)).squeeze(dim=-1)
### END YOUR CODE

