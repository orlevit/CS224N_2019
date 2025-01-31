#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx         = vocab.char2id['<pad>']
        char_embed_size       = 50
        max_word_len          = 21
        dropout_rate          = 0.3

        self.embed_size       = embed_size
        self.char_embedding   = nn.Embedding(len(vocab.char2id), char_embed_size, pad_token_idx)
        self.CNN              = CNN(f = embed_size, e_char_size = char_embed_size, max_word_len = max_word_len)
        self.Highway          = Highway(embed_word_size=embed_size)
        self.dropout          = nn.Dropout(p=dropout_rate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        all_X_word_emb = []
        # Batchs of the input of size (batch_size, max_word_length)
        for X_padded in input:
            X_emb      = self.char_embedding(X_padded)
            X_reshaped = torch.transpose(X_emb, dim0=-1, dim1=-2)
            X_conv_out = self.CNN(X_reshaped)
            X_highway  = self.Highway(X_conv_out)
            X_word_emb = self.dropout(X_highway)
            all_X_word_emb.append(X_word_emb)

        X_word_emb = torch.stack(all_X_word_emb)
        return (X_word_emb)
        ### END YOUR CODE

