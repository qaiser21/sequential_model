import numpy as np
import random


# Dataset and preprocessing

data = open('dinos.txt','r').read()
data = data.lower()
chars = list(set(data))

data_size = len(data)
vocab_size = len(chars)

print("total number of character is %d, and total number of unique character is %d ".%(data_size, vocab_size))


# mapping character to index, index to character
char_to_ix = {ch:i for i,ch in enumerate(sorted(chars))}
ix_to_char = {i:ch for i,ch in enumerate(sorted(chars))}


def clip(gradients, maxvalue):
