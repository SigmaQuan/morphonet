# -*- coding: utf-8 -*-
import numpy as np
import pdb
import os
import sys
current_py_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_py_file))))

from configs.globals import PROJECT_FOLDER
DATA_FOLDER = PROJECT_FOLDER + "/data/word_analogy/"

FILE = "analogy.txt"


def analogy(pairs, embeddings, word_dic, word_set):
    total = len(pairs)
    reverse_dict = dict(zip(word_dic.values(), word_dic.keys()))
    in_dict_cnt = 0
    predict_cnt = 0
    # print('dictionary_lengh ', len(word_dic))
    for pair in pairs:
        in_dict = True
        for i in range(len(pair)):
            pair[i] = pair[i].decode('utf-8')
            in_dict = in_dict and (pair[i] in word_set)
        if in_dict:
            in_dict_cnt += 1
            predict_cnt += predict_word(pair[0], pair[1], pair[2], pair[3], embeddings, word_dic, reverse_dict)

    return total, in_dict_cnt, predict_cnt


def predict_word(w1, w2, w3, w4, embeddings, word_dic, reverse_dict):
    # return the index of predicted word
    id1 = word_dic[w1]
    id2 = word_dic[w2]
    id3 = word_dic[w3]
    # reverse_dict = dict(zip(word_dic.values(), word_dic.keys()))
    pattern = embeddings[id2] - embeddings[id1] + embeddings[id3]
    pattern /= np.linalg.norm(pattern)
    sim = embeddings.dot(pattern.T)
    sim[id1] = sim[id2] = sim[id3] = -1  # remove the input words
    predict_index = np.argmax(sim)
    id4 = word_dic[w4]
    if predict_index == id4:
        # print("WRIGHT: ({}: {}), ({}: {}), {}".format(w1, w2, w3, w4, reverse_dict[predict_index]))
        return 1
    else:
        pass
        # print(w1)
        # print(w2)
        # print(w3)
        # print(w4)
        # print(reverse_dict[predict_index])
        # print("ERROR: ({}: {}), ({}: {}), {}".format(w1, w2, w3, w4, reverse_dict[predict_index]))
        # pdb.set_trace()

    return 0


def get_total_words():
    file_path = DATA_FOLDER + FILE
    total, _, _, _ = read_read_word_analogy(file_path)

    words = []
    for pair in total:
        for i in range(4):
            if pair[i] not in words:
                words.append(pair[i])

    return words


def read_read_word_analogy(file_path):
    file_stream = open(file_path, 'r')

    capital = []
    state = []
    family = []
    cnt = 0
    for line in file_stream:
        pair = line.split()
        if pair[0] == ':':
            cnt = cnt + 1
            continue
        if cnt == 1:
            capital.append(pair)
        elif cnt == 2:
            state.append(pair)
        else:
            family.append(pair)
    file_stream.close()

    total = capital + state + family

    return total, capital, state, family
