import os
import time
import pytest
from configs.globals import PROJECT_FOLDER
from morphonets.datasets import word_similarity_computation as wsc

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

FOLDER = PROJECT_FOLDER + \
         time.strftime('/cache/datasets/word_similarity_computation/%Y-%m-%d-%H-%M-%S/')


def test_get_total_characters():
    total_characters = wsc.get_total_characters(wsc.get_total_words())

    print("Total characters in the two files.")
    for i, character in enumerate(total_characters):
        print("{}, {}".format(i, character))


def test_get_total_words():
    total_words = wsc.get_total_words()

    print("Total words in the two files.")
    for i, word in enumerate(total_words):
        print(u"{}, {}".format(i, word))


def test_get_words():
    file_paths = []
    file_paths.append(wsc.DATA_FOLDER + wsc.FILE_1)
    file_paths.append(wsc.DATA_FOLDER + wsc.FILE_2)

    for file_path in file_paths:
        print("Get words from file: \n{}".format(file_path))
        words = wsc.get_words(file_path)
        for i, word in enumerate(words):
            print(u"{}, {}".format(i, word))


def test_read_word_pairs():
    file_paths = []
    file_paths.append(wsc.DATA_FOLDER + wsc.FILE_1)
    file_paths.append(wsc.DATA_FOLDER + wsc.FILE_2)

    for file_path in file_paths:
        print("Get words from file: \n{}".format(file_path))
        word_pairs = wsc.read_word_pair(file_path)
        for pair in word_pairs:
            print(u"{}, {}, {}".format(pair[0], pair[1], pair[2]))


def test(folder=FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/word_similarity_computation.log' % folder
    sys.stdout = open(log_file, 'a')

    # get characters
    test_get_total_characters()

    # get words
    test_get_total_words()

    # get words
    test_get_words()

    # get word pair and its similarity
    test_read_word_pairs()

    sys.stdout.close_board()
    sys.stdout = sys_stdout


if __name__ == '__main__':
    pytest.main([__file__])
