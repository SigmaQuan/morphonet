import os
import time
import pytest
import numpy as np

from morphonets.knowledge import cc_phonology
from morphonets.datasets import zhwiki_corpus
from morphonets.datasets import word_analogy_reasoning
from morphonets.datasets import word_similarity_computation
from configs.globals import PROJECT_FOLDER

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

DATA_FOLDER = PROJECT_FOLDER + "/data/corpus/"

FOLDER = PROJECT_FOLDER + \
         time.strftime('/cache/corpus/%Y-%m-%d-%H-%M-%S/')


def test_generator(folder=FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/corpus_test_generator.log' % folder
    sys.stdout = open(log_file, 'a')

    print("generate samples:")
    generator = zhwiki_corpus.generator()

    # get ids.
    initials, _, _, _, finals, _, tones, _ = cc_phonology.get_initial_final_tone_sets()
    id_pinyins = cc_phonology.get_id_pinyin_dic()
    id_initials, id_finals, id_tones = cc_phonology.get_id_other_dic()

    # get word
    word_dic, id_word_dic = zhwiki_corpus.get_word_id_dictionaries()

    max_batch = 20
    for i, batch_samples in enumerate(generator):
        print("batch number %d: " % i)
        (input_sequences, target_sequences_character,
         target_sequences_pinyin, target_sequences_initial,
         target_sequences_final, target_sequences_tone) = batch_samples

        for j in range(len(input_sequences)):
            print('input_sequences')
            print(input_sequences[j])
            for k in range(len(input_sequences[j])):
                print(id_word_dic[input_sequences[j][k]])

            print('target_sequences_character')
            print(target_sequences_character[j])
            for k in range(len(target_sequences_character[j])):
                print(id_word_dic[np.argmax(target_sequences_character[j][k])])

            print('target_sequences_pinyin')
            print(target_sequences_pinyin[j])
            for k in range(len(target_sequences_pinyin[j])):
                print(id_pinyins[np.argmax(target_sequences_pinyin[j][k])])

            print('target_sequences_initial')
            print(target_sequences_initial[j])
            for k in range(len(target_sequences_initial[j])):
                print(id_initials[np.argmax(target_sequences_initial[j][k])])

            print('target_sequences_final')
            print(target_sequences_final[j])
            for k in range(len(target_sequences_final[j])):
                print(id_finals[np.argmax(target_sequences_final[j][k])])

            print('target_sequences_tone')
            print(target_sequences_tone[j])
            for k in range(len(target_sequences_tone[j])):
                print(id_tones[np.argmax(target_sequences_tone[j][k])])

            print('\n')

        print('\n\n')

        if i >= max_batch:
            break

    sys.stdout.close_board()
    sys.stdout = sys_stdout


def test_get_words_characters(folder=FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/corpus_characters_words.log' % folder
    sys.stdout = open(log_file, 'a')

    print("characters:")
    character_dic, id_character_dic = zhwiki_corpus.get_character_id_dictionaries()
    for id in sorted(id_character_dic.keys()):
        print("{}, {}".format(id, id_character_dic[id]))

    print("words:")
    word_dic, id_word_dic = zhwiki_corpus.get_word_id_dictionaries()
    for id in sorted(id_word_dic.keys()):
        print("{}, {}".format(id, id_word_dic[id]))

    sys.stdout.close_board()
    sys.stdout = sys_stdout


def test_get_formalized_statistics(folder=FOLDER, data_folder=DATA_FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/corpus_formalized_statistics.log' % folder
    sys.stdout = open(log_file, 'a')

    # zhwiki_corpus.get_formalized_statistics()

    word_count, character_count, word_idx, character_idx, tokens_number, \
    characters_number = zhwiki_corpus.get_statistics(zhwiki_corpus.ZHWIKI_FOLDER_CACHE)

    print("total words: ", len(word_count))
    print("word distribution: ")
    totals_w = sum(word_count.values())
    word_count_list = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
    # totals = sum(word_count.values())
    word_distribution = []
    count = 0
    i = 0
    for item in word_count_list:
        i += 1
        count += item[1]
        word_distribution.append(count / float(totals_w))
        if item[1] >= 1:
            print("{:>11}, {:>8}, {:<15}, {}".format(
                i, item[1], count / float(totals_w), item[0].encode('utf-8')))
    # hist(word_distribution)

    print("total characters: ", len(character_count))
    print("word distribution: ")
    totals_c = sum(character_count.values())
    character_count_list = sorted(character_count.items(), key=lambda d: d[1], reverse=True)
    character_distribution = []
    count = 0
    i = 0
    for item in character_count_list:
        i += 1
        count += item[1]
        character_distribution.append(count / float(totals_c))
        if item[1] >= 1:
            print("{:>11}, {:>8}, {:<15}, {}".format(
                i, item[1], count / float(totals_c), item[0].encode('utf-8')))

    total_words_1 = word_similarity_computation.get_total_words()
    total_characters_1 = word_similarity_computation.get_total_characters(total_words_1)

    total_words_2 = word_analogy_reasoning.get_total_words()
    total_characters_2 = word_similarity_computation.get_total_characters(total_words_2)

    total_words = set(total_words_1) | set(total_words_2)
    total_characters = set(total_characters_1) | set(total_characters_2)

    i = 0
    count = 0
    print(len(total_words))
    for item in total_words:
        item = item.encode('utf-8')
        if item in word_count.keys():
            i += 1
            count += word_count[item]
            print("{:>11}, {:>8}, {:<15}, {}".format(
                i, word_count[item], count / float(totals_w), item))
        else:
            print("******: {}".format(item))

    i = 0
    count = 0
    print(len(total_characters))
    for item in total_characters:
        item = item.encode('utf-8')
        if item in character_count.keys():
            i += 1
            count += character_count[item]
            print("{:>11}, {:>8}, {:<15}, {}".format(
                i, character_count[item], count / float(totals_c), item))
        else:
            print("******: {}".format(item))

    sys.stdout.close_board()
    sys.stdout = sys_stdout


def test_statistics(folder=FOLDER, data_folder=DATA_FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/corpus_statistics.log' % folder
    sys.stdout = open(log_file, 'a')

    # word_count, character_count, \
    # word_idx, character_idx, \
    # tokens_number, characters_number

    word_count, character_count, \
    word_idx, character_idx, \
    tokens_number, characters_number = zhwiki_corpus.get_statistics()
    print("******")
    print("total word size %7d, character size %5d, tokens %d , characters %d." %
          (len(word_count), len(character_count), tokens_number, characters_number))

    print("******")
    print("word count and index :")
    for key in word_count.keys():
        print("{:>8}, {:>8}: {}".format(word_idx[key], word_count[key], key.encode('utf-8')))
        # print("%s: %8d, %8d" % (key.encode('utf-8'), word_count[key], word_idx[key]))
    print("*******\n\n\n\n")

    print("******")
    print("character count and index :")
    for key in character_count.keys():
        print("{}: {:>8}, {:>8}".format(key.encode('utf-8'), character_count[key], character_idx[key]))
        # print("%s: %8d, %8d" % (key.encode('utf-8'), character_count[key], character_idx[key]))
    print("*******\n\n\n\n")

    sys.stdout.close_board()
    sys.stdout = sys_stdout


if __name__ == '__main__':
    pytest.main([__file__])
