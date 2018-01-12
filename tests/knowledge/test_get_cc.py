# -*- coding: utf-8 -*-
"""
Test Chinese characters come from Universal Standard Chinese Character
List which released by Chinese State Council at 2013.
"""
import os
import sys
import time
import pytest

from configs.globals import PROJECT_FOLDER
import morphonets.knowledge.cc_level_1 as level_1_3500_2013
import morphonets.knowledge.cc_level_2 as level_2_3000_2013

FOLDER = PROJECT_FOLDER + \
         time.strftime('/logs/knowledge/cc_get/%Y-%m-%d-%H-%M-%S/')


def test_get_level_1_characters(folder=FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/cc_get_1.log' % folder
    sys.stdout = open(log_file, 'a')

    level_1_3500_characters = level_1_3500_2013.get_characters()
    level_1_size = len(level_1_3500_characters)
    print("There are total %d level 1 Chinese characters:" % level_1_size)
    for i, character in enumerate(level_1_3500_characters):
        print("{:>3}, {}".format(i, character.encode('utf-8')))

    print(level_1_3500_characters[0].encode('utf-8'))

    sys.stdout.close_board()
    sys.stdout = sys_stdout


def test_get_level_2_characters(folder=FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/cc_get_2.log' % folder
    sys.stdout = open(log_file, 'a')

    level_2_3000_characters = level_2_3000_2013.get_characters()
    level_2_size = len(level_2_3000_characters)
    print("There are total %d level 2 Chinese characters:" % level_2_size)
    for i, character in enumerate(level_2_3000_characters):
        print("{:>3}, {}".format(i, character.encode('utf-8')))

    print("688, 689, 690")
    print((level_2_3000_characters[688]+level_2_3000_characters[689]+level_2_3000_characters[690]).encode('utf-8'))
    print("688")
    print((level_2_3000_characters[688]).encode('utf-8'))
    print("689")
    print((level_2_3000_characters[689]).encode('utf-8'))
    print("690")
    print((level_2_3000_characters[690]).encode('utf-8'))

    print("2488, 2489, 2490")
    print((level_2_3000_characters[2488]+level_2_3000_characters[2489]+level_2_3000_characters[2490]).encode('utf-8'))
    print("2488")
    print((level_2_3000_characters[2488]).encode('utf-8'))
    print("2489")
    print((level_2_3000_characters[2489]).encode('utf-8'))
    print("2490")
    print((level_2_3000_characters[2490]).encode('utf-8'))

    sys.stdout.close_board()
    sys.stdout = sys_stdout


if __name__ == '__main__':
    pytest.main([__file__])
