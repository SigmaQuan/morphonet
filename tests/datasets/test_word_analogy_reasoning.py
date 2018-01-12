import os
import time
import pytest

from configs.globals import PROJECT_FOLDER
from morphonets.datasets import word_analogy_reasoning as war
from morphonets.datasets.word_similarity_computation import get_total_characters

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

FOLDER = PROJECT_FOLDER + \
         time.strftime('/cache/datasets/word_analogy_reasoning/%Y-%m-%d-%H-%M-%S/')


def test_get_total_characters():
    total_characters = get_total_characters(war.get_total_words())

    print("Total characters in the file.")
    for i, character in enumerate(total_characters):
        print("{}, {}".format(i, character))


def test_get_total_words():
    total_words = war.get_total_words()

    print("Total words in the file.")
    for i, word in enumerate(total_words):
        print(u"{}, {}".format(i, word))


def test_read_word_pairs():
    file_path = war.DATA_FOLDER + war.FILE

    print("Get words from file: \n{}".format(file_path))
    total, capital, state, family = war.read_read_word_analogy(file_path)
    print('total')
    for pair in total:
        print(u"{}, {}, {}, {}".format(pair[0], pair[1], pair[2], pair[3]))

    print('capital')
    for pair in capital:
        print(u"{}, {}, {}, {}".format(pair[0], pair[1], pair[2], pair[3]))

    print('state')
    for pair in state:
        print(u"{}, {}, {}, {}".format(pair[0], pair[1], pair[2], pair[3]))

    print('family')
    for pair in family:
        print(u"{}, {}, {}, {}".format(pair[0], pair[1], pair[2], pair[3]))


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

    # get word pair and its similarity
    test_read_word_pairs()

    sys.stdout.close_board()
    sys.stdout = sys_stdout


if __name__ == '__main__':
    pytest.main([__file__])
