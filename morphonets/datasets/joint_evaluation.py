# -*- coding: utf-8 -*-
import numpy as np
import time
import sys
import os
# import model
# import config
from word_similarity_computation import read_word_pair, evaluation
import word_analogy_reasoning as war
import word_similarity_computation as wsc
import zhwiki_corpus


def run_wsc_1(word_set, word_dic, word_embedding):
    print(time.strftime('%Y-%m-%d %H:%M:%S') +
          " Test {}".format(wsc.FILE_1))
    word_pair = read_word_pair(wsc.DATA_FOLDER + wsc.FILE_1)
    rho, pval = evaluation(word_pair, word_set, word_dic, word_embedding)
    print(time.strftime('%Y-%m-%d %H:%M:%S') +
          " Score is rho {}, pval {}".format(rho, pval))
    return rho, pval


def run_wsc_2(word_set, word_dic, word_embedding):
    print(time.strftime('%Y-%m-%d %H:%M:%S') +
          " Test {}".format(wsc.FILE_2))
    word_pair = read_word_pair(wsc.DATA_FOLDER + wsc.FILE_2)
    rho, pval = evaluation(word_pair, word_set, word_dic, word_embedding)
    print(time.strftime('%Y-%m-%d %H:%M:%S') +
          " Score is rho {}, pval {}".format(rho, pval))
    return rho, pval


def run_war_once(word_pair, word_embedding, word_dic, word_set, part_name):
    total, in_dict_cnt, predict_cnt = war.analogy(
        word_pair, word_embedding, word_dic, word_set)
    print(time.strftime('%Y-%m-%d %H:%M:%S') +
          " Score in {} total {}, "
          "in_dict_cnt {}, predict_cnt {}".format(part_name,
              total, in_dict_cnt, predict_cnt))
    return predict_cnt / (in_dict_cnt + 0.0)


def get_embeddings(total_embedding, word_dic, word_set, words):
    # embedding_part = []
    word_set_part = []
    word_dict_part = {}

    num_words = 0
    for word in words:
        word_ = word.decode('utf-8')
        if word_ in word_set:
            word_dict_part[word_] = num_words
            # id = word_dic[word_]
            word_set_part.append(word_)
            # embedding_part.append(total_embedding[id])
            num_words += 1

    embedding_part = np.ndarray((num_words, len(total_embedding[0])), dtype=float)
    num_words = 0
    for word in words:
        word_ = word.decode('utf-8')
        if word_ in word_set:
            id = word_dic[word_]
            embedding_part[num_words] = total_embedding[id]
            num_words += 1

    word_set_part = set(word_set_part)

    # print(type(total_embedding))
    # print(type(embedding_part))

    return embedding_part, word_set_part, word_dict_part


def general_measure(word_embedding):
# def general_measure(embeding_model):
    # get total word embedding
    # word_embedding = model.get_embedding_layer(embeding_model)

    # print(len(word_embedding))
    # for i in range(len(word_embedding)):
    #     print(word_embedding[i])
    word_dic, _ = zhwiki_corpus.get_word_id_dictionaries()
    word_set = set(word_dic.keys())

    # get
    words_s = wsc.get_total_words()
    word_embedding_s, word_set_s, word_dict_s = get_embeddings(
        word_embedding, word_dic, word_set, words_s)

    words_a = war.get_total_words()
    word_embedding_a, word_set_a, word_dict_a = get_embeddings(
        word_embedding, word_dic, word_set, words_a)

    # begin testing
    print(time.strftime('\n%Y-%m-%d %H:%M:%S') + " ")
    # run_wsc_1(word_set_s, word_dict_s, word_embedding_s)
    # run_wsc_2(word_set_s, word_dict_s, word_embedding_s)
    loss_wsc_rho_1, loss_wsc_pval_1 = run_wsc_1(word_set_s, word_dict_s, word_embedding_s)
    loss_wsc_rho_2, loss_wsc_pval_2 = run_wsc_2(word_set_s, word_dict_s, word_embedding_s)

    print(time.strftime('%Y-%m-%d %H:%M:%S') + " Test {}".format(war.FILE))
    total, capital, state, family = war.read_read_word_analogy(
        war.DATA_FOLDER + war.FILE)
    # acc_war_total = run_war_once(total, word_embedding_a, word_dict_a, word_set_a, "total")
    loss_war_capital = run_war_once(capital, word_embedding_a, word_dict_a, word_set_a, "capital")
    loss_war_state = run_war_once(state, word_embedding_a, word_dict_a, word_set_a, "state")
    loss_war_family = run_war_once(family, word_embedding_a, word_dict_a, word_set_a, "family")
    loss_war_total = (len(capital) * loss_war_capital +
                      len(state) * loss_war_state +
                      len(family) * loss_war_family) / len(total)

    return loss_wsc_rho_1, loss_wsc_pval_1, \
           loss_wsc_rho_2, loss_wsc_pval_2, \
           loss_war_total, loss_war_capital, loss_war_state, loss_war_family


# def run():
#     start_time = time.time()
#
#     # dump hyper parameters settings
#     print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Hyper-parameters setting')
#     # get configuration
#     args = config.get()
#
#     if os.path.isfile(args.model_file_train):
#         print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Load model from file...')
#         embeding_model = load_model(args.model_file_test)
#     else:
#         # get embedding model
#         print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Build model...')
#         embeding_model = model.get(args)
#
#     general_measure(embeding_model)
#
#     # close_board windows
#     # dashboard.close_board()
#     print("task took %.3fs" % (float(time.time()) - start_time))


# log_every = True
# # log_every = False
# # create log file
# if log_every:
#     sys_stdout = sys.stdout
#     log_file = '%s/testing_embedding.log' % configs.FOLDER
#     sys.stdout = open(log_file, 'a')
#
# # testing
# # run()
#
# # create log file
# if log_every:
#     sys.stdout.close_board()
#     sys.stdout = sys_stdout
