# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.stats import spearmanr
import sys
current_py_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_py_file))))

from configs.globals import PROJECT_FOLDER
DATA_FOLDER = PROJECT_FOLDER + "/data/word_similarity/"

FILE_1 = "240.txt"
FILE_2 = "297.txt"


def get_total_characters(total_words):
    # total_words = get_total_words()

    total_characters = []
    for word in total_words:
        word = word.decode('utf-8')
        word_length = len(word)
        for i in range(word_length):
            character = word[i]
            if character not in total_characters:
                total_characters.append(character)

    return total_characters


def get_total_words():
    file_paths = []
    file_paths.append(DATA_FOLDER + FILE_1)
    file_paths.append(DATA_FOLDER + FILE_2)

    total_words = []
    for file_path in file_paths:
        words = get_words(file_path)
        for word in words:
            if word not in total_words:
                total_words.append(word)

    return total_words


def get_words(file_path):
    word_pairs = read_word_pair(file_path)

    words = []
    for pair in word_pairs:
        if pair[0] not in words:
            words.append(pair[0])
        if pair[1] not in words:
            words.append(pair[1])

    return words


def read_word_pair(file_path):
    file_stream = open(file_path, 'r')

    word_pairs = []
    for line in file_stream:
        pair = line.split()
        pair[2] = float(pair[2])
        word_pairs.append(pair)

    file_stream.close()

    return word_pairs


def evaluation(word_pairs, word_set, word_dic, word_embedding):
    human_similarity = []
    vector_similarity = []
    cnt = 0
    total = len(word_pairs)
    for pair in word_pairs:
        w1 = pair[0]
        w2 = pair[1]
        w1 = w1.decode('utf-8')
        w2 = w2.decode('utf-8')
        if w1 in word_set and w2 in word_set:
            cnt += 1
            id1 = word_dic[w1]
            id2 = word_dic[w2]
            vsim = word_embedding[id1].dot(word_embedding[id2].T) / (
            np.linalg.norm(word_embedding[id1]) * np.linalg.norm(word_embedding[id2]))
            human_similarity.append(pair[2])
            vector_similarity.append(vsim)
            # vector_similarity.append(2*pair[2]+np.random.random())
    # print(cnt, ' word word_pairs appered in the training dictionary , total word word_pairs ', total)
    # print("{} word word_pairs appered in the training dictionary , total word word_pairs {}".format(cnt, total))
    rho, pval = spearmanr(human_similarity, vector_similarity)

    return rho, pval
