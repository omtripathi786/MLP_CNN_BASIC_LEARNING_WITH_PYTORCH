import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def read_text():
    with open('data/anna.txt', 'r') as f:
        text = f.read()
    return text


def do_tokenization(text_str):
    chars = tuple(set(text_str))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char2int[ch] for ch in text_str])
    return encoded


def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


if __name__ == '__main__':
    text = read_text()
    encoded = do_tokenization(text)
    print('yessssss')
