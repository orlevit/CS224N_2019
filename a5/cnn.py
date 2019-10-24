#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    """ CNN computation
    """
    def __init__(self, f, e_char_size, max_word_len, k = 5):
        """ Init CNN Model.
        :param f (int): number of filter
        :param e_char_size (int): Embedding word size (dimensionality)
        :param k (int): number of kernels
        :param max_word_len (int): maximum length of a word
        """
        super(CNN, self).__init__()
        self.f            = f
        self.e_char_size  = e_char_size
        self.k            = k
        self.max_word_len = max_word_len

        self.conv1d = nn.Conv1d(in_channels=self.e_char_size, out_channels=self.f,
                                kernel_size=self.k, bias=True)

        self.max_pool = nn.MaxPool1d(self.max_word_len - self.k + 1)

    def forward(self, input):
        """
        Take a mini batch of character embedding of each word, compute word embedding
        :param input (Tensor): shape (batch_size, char_embed_size, max_word_length)
        :return (Tensor): shape (batch_size, word_embed_size), word embedding of each word in batch
        """
        x_conv     = self.conv1d(input)  # (batch_size, word_embed_size, max_word_length - kernel_size + 1)
        x_conv_out = self.max_pool(F.relu(x_conv)).squeeze(-1)  # (batch_size, word_embed_size)
        return x_conv_out

### END YOUR CODE