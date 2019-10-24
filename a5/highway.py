#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """ Highway computation
    """
    def __init__(self, embed_word_size):
        """ Init Highway Model.
        :param embed_size (int): Embedding word size (dimensionality)
        """
        super(Highway, self).__init__()
        self.embed_word_size = embed_word_size
        self.proj = nn.Linear(self.embed_word_size , self.embed_word_size , bias=True)
        self.gate = nn.Linear(self.embed_word_size , self.embed_word_size , bias=True)

    def forward(self, x_conv_out):
        """ Compute highway
        :param x_conv_out: input into highway
        """
        x_proj    = F.relu(self.proj(x_conv_out))
        x_gate    = F.sigmoid(self.gate(x_conv_out))
        x_highway = torch.mul(x_proj,x_gate) + torch.mul(x_conv_out,(1- x_gate))

        return x_highway

### END YOUR CODE 

