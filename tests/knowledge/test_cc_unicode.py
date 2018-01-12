# -*- coding: utf-8 -*-
import morphonets.knowledge.cc_unicode as cc_unicode
import json
import pytest
import os

import sys
import time
from configs.globals import PROJECT_FOLDER


DATA_FOLDER = PROJECT_FOLDER + "/data/knowledge/"

FOLDER = PROJECT_FOLDER + \
         time.strftime('/logs/knowledge/cc_unicode/%Y-%m-%d-%H-%M-%S/')


def test_cc_unicode(folder=FOLDER, data_folder=DATA_FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/cc_unicode.log' % folder
    sys.stdout = open(log_file, 'a')

    unicode_dic = cc_unicode.get()
    print(len(unicode_dic))
    # special_character = u"𠳐"
    # print(unicode_dic[special_character[0] + special_character[1]])
    # print(unicode_dic[special_character])
    print(unicode_dic[u'\U00025ed7'])
    print(unicode_dic)
    json.dump(unicode_dic, open(data_folder+'unicode_of_chinese_character.txt', 'w'))
    cc_unicodes = json.load(open(data_folder+'unicode_of_chinese_character.txt', 'r'))
    print(cc_unicodes)

    sys.stdout.close_board()
    sys.stdout = sys_stdout


if __name__ == '__main__':
    pytest.main([__file__])
