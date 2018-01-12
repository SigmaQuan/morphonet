# -*- coding: utf-8 -*-
import morphonets.knowledge.cc_phonology as cc_phonology
# import ..knowledge.cc_phonology as cc_phonology
import pytest
import os
import time
from configs.globals import PROJECT_FOLDER

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


DATA_FOLDER = PROJECT_FOLDER + "/data/knowledge/"

FOLDER = PROJECT_FOLDER + \
         time.strftime('/logs/knowledge/cc_phonology/%Y-%m-%d-%H-%M-%S/')


def test_cc_phonology_cc_pinyin(folder=FOLDER, data_folder=DATA_FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/cc_phonology_cc_pinyin.log' % folder
    sys.stdout = open(log_file, 'a')

    character_pinyin_dic = cc_phonology.get_pinyins_of_character()

    for (character, pinyins) in character_pinyin_dic.items():
        pinyin = u""
        for py in pinyins:
            pinyin += py
            pinyin += u", "

        print(u"{}: {}".format(character, pinyin))

    sys.stdout.close_board()
    sys.stdout = sys_stdout


def test_cc_phonology_grained_pinyin(folder=FOLDER, data_folder=DATA_FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/cc_phonology_grained_pinyin.log' % folder
    sys.stdout = open(log_file, 'a')

    phonology_dic = cc_phonology.get_knowledge_of_fine_grained_pinyin()

    print(u"{:>4}, {:>9}, {:>8}, {:>2}, {:>8}, {:>2}, {:>8}, {:>2}, {}".format(
        "ID", "pinyin", "initial", "ID", "Final", "ID", "Tone", "ID", "Chinese characters"))
    for i, key in enumerate(sorted(phonology_dic.keys())):
        tmp = u""
        for character in phonology_dic[key]["Characters"]:
            tmp += character.encode('utf-8')

        print(u"{:>4}, {:>9}, {:>8}, {:>2}, {:>8}, {:>2}, {:>8}, {:>2}, {}".format(i, key,
            phonology_dic[key]["Initial"], phonology_dic[key]["Initial_ID"],
            phonology_dic[key]["Final"], phonology_dic[key]["Final_ID"],
            phonology_dic[key]["Tone"], phonology_dic[key]["Tone_ID"], tmp))

    sys.stdout.close_board()
    sys.stdout = sys_stdout


if __name__ == '__main__':
    pytest.main([__file__])
