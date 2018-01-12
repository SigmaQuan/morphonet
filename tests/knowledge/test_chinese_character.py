# -*- coding: utf-8 -*-
import os
import sys
import time
import pytest
from configs.globals import PROJECT_FOLDER
import morphonets.knowledge.chinese_character as chinese_character


DATA_FOLDER = PROJECT_FOLDER + "/data/knowledge/"

FOLDER = PROJECT_FOLDER + \
         time.strftime('/logs/knowledge/cc/%Y-%m-%d-%H-%M-%S/')


def test_get_pinyin_and_radical():

    dic = chinese_character.get_pinyin_and_radical()

    for (character, info) in dic.items():
        print(u"character: " + character + u" unicode: " + info[u"Unicode"] + u" URL: " + info[u"URL"])

        pinyins = u"\tpingyin: "
        # print u"\tpingyin: "
        for item in info[u"PinYin"]:
            pinyins += repr(item) + u", "
            # print(u"\t\t" + repr(item))
        print(pinyins)

        radicals = u"\tradical: "
        radicals += u"head: " + info[u"Radical"][u"Head"]
        radicals += u"; other: " + info[u"Radical"][u"Other"]
        print(radicals)
        # print u"\tradical: "
        # print u"\t\thead: " + info[u"Radical"][u"Head"]
        # print u"\t\thead link: " + info[u"Radical"][u"HeadLink"]
        # print u"\t\tother: " + info[u"Radical"][u"Other"]
        # print u"\t\tother link: " + info[u"Radical"][u"OtherLink"]

        structures = u"\tstructures: "
        structures += u"tpye: " + info[u"Structure"][u"Type"]
        structures += u"; content: " + info[u"Structure"][u"Content"]
        print(structures)
        # print u"\tstructure: "
        # print u"\t\ttpye: " + info[u"Structure"][u"Type"]
        # print u"\t\tcontent: " + info[u"Structure"][u"Content"]
        # print u"\t\tlink: " + info[u"Structure"][u"Link"]

        strokes = u"\tstrokes: "
        strokes += u"number: " + info[u"Stroke"][u"Number"]
        strokes += u"; write: " + info[u"Stroke"][u"Write"]
        print(strokes)
        # print u"\tstroke: "
        # print u"\t\tnumber: " + info[u"Stroke"][u"Number"]
        # print u"\t\twrite: " + info[u"Stroke"][u"Write"]

#
# def test_cc():
#     dic = chinese_character.get_pinyin_and_radical()
#     print(dic)


def test(folder=FOLDER, data_folder=DATA_FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/cc_unicode.log' % folder
    sys.stdout = open(log_file, 'a')

    test_get_pinyin_and_radical()

    sys.stdout.close_board()
    sys.stdout = sys_stdout


if __name__ == '__main__':
    pytest.main([__file__])
