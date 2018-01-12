# -*- coding: utf-8 -*-
"""
Help classes and functions.
"""

import numpy as np
import random
import time

# from keras.callbacks import Callback
# from math import exp
# import matplotlib.pyplot as plt
# import numpy as np
# from keras import losses
# from keras import backend as K


def initialize_random_seed():
    seed = int(time.time())
    np.random.seed(seed)
    random.seed(seed)


def generate_random_binomial_(row, col):
    return np.random.binomial(
        1, 0.5, (row, col)).astype(np.uint8)


def check_contain_chinese(check_str):
    for ch in check_str.decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def pre_process_words(one_line):
    for i in range(len(one_line)):
        word = one_line[i]
        one_line[i] = pre_process(word)

    return one_line


def pre_process(one_word):

    if len(one_word) >= 2 and one_word[-1] == u'\n':
        print(one_word)
        word = one_word.replace(u'\n', u'')
        return word

    return one_word
