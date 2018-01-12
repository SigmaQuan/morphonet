# -*- coding: utf-8 -*-
import os
import json
from chinese_character import get_pinyin_and_radical
import cc_level_1
import cc_level_2

import os
import sys
current_py_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_py_file))))
from configs.globals import PROJECT_FOLDER

DATA_FOLDER = PROJECT_FOLDER + "/data/knowledge/"


def get_pinyins_of_character():
    character_pinyin_dic = {}

    dic = get_pinyin_and_radical()

    characters_level_1 = cc_level_1.get_characters()
    characters_level_2 = cc_level_2.get_characters()
    characters = characters_level_1 + characters_level_2

    for character in characters:
        character_pinyin_dic[character] = dic[character][u"PinYin"]

    return character_pinyin_dic


def get_knowledge_of_fine_grained_pinyin():
    pinyin_dic = initial_pinyin_dic()

    initials, initials_1, initials_2, initials_ID, \
    finals, finals_ID, \
    tones, tones_ID = get_initial_final_tone_sets()

    size = 0
    for key in pinyin_dic.keys():
        # print(size)
        # size += 1
        # print(key)
        # print(pinyin_dic[key])
        set_initial_final_tone(
            key, pinyin_dic[key],
            initials_1, initials_2, initials_ID,
            finals, finals_ID, tones, tones_ID)
        # print(pinyin_dic[key])

    # print(pinyin_dic)

    return pinyin_dic


def get_pinyin_id_dic():
    pinyin_id_dic = {}
    pinyin_dic = get_knowledge_of_fine_grained_pinyin()

    for i, key in enumerate(sorted(pinyin_dic.keys())):
        pinyin_id_dic[key] = i

    return pinyin_id_dic


def get_id_pinyin_dic():
    pinyin_id_dic = get_pinyin_id_dic()
    id_pinyins = {}
    for item in sorted(pinyin_id_dic.items()):
        id_pinyins[item[1]] = item[0]

    return id_pinyins


def get_id_other_dic():
    _, _, _, initials_ID, _, finals_ID, _, tones_ID = get_initial_final_tone_sets()

    id_initials = {}
    for item in sorted(initials_ID.items()):
        id_initials[item[1]] = item[0]

    id_finals = {}
    for item in sorted(finals_ID.items()):
        id_finals[item[1]] = item[0]

    id_tones = tones_ID
    # for item in sorted(tones_ID.items()):
    #     id_tones[item[1]] = item[0]

    return id_initials, id_finals, id_tones


def initial_pinyin_dic():
    dump_file = DATA_FOLDER + 'cc_dictionary_filled.txt'
    if os.path.exists(dump_file):
        dic = json.load(open(dump_file, 'r'))
    else:
        pass

    none_pinyin_info = {}
    none_pinyin_info[u"Characters"] = []
    none_pinyin_info[u"Initial"] = "NONE"
    none_pinyin_info[u"Final"] = "NONE"
    none_pinyin_info[u"Initial_ID"] = 0
    none_pinyin_info[u"Final_ID"] = 0
    none_pinyin_info[u"Tone"] = "UN_KNOWN"
    none_pinyin_info[u"Tone_ID"] = 0
    un_known_pinyin_info = {}
    un_known_pinyin_info[u"Characters"] = []
    un_known_pinyin_info[u"Initial"] = "UN_KNOWN"
    un_known_pinyin_info[u"Final"] = "UN_KNOWN"
    un_known_pinyin_info[u"Initial_ID"] = 24
    un_known_pinyin_info[u"Final_ID"] = 42
    un_known_pinyin_info[u"Tone_ID"] = 5
    un_known_pinyin_info[u"Tone"] = "UN_KNOWN"
    pinyin_dic = {"NONE0": none_pinyin_info,
                  "SPECIAL_SYMBOL": un_known_pinyin_info,
                  "UN_KNOWN_CHAR": un_known_pinyin_info}

    for (character, info) in dic.items():
        # print(u"character: " + character + u" unicode: " + info[u"Unicode"] + u" URL: " + info[u"URL"])
        # pinyins = u"\tpingyin: "
        for item in info[u"PinYin"]:
            # pinyins += repr(item) + u", "
            if item in pinyin_dic.keys():
                pinyin_dic[item][u"Characters"].append(character)
            else:
                one_pinyin_info = {}
                one_pinyin_info[u"Characters"] = []
                one_pinyin_info[u"Characters"].append(character)
                one_pinyin_info[u"Initial"] = 'UN_KNOWN'
                one_pinyin_info[u"Initial_ID"] = 24
                one_pinyin_info[u"Final"] = 'UN_KNOWN'
                one_pinyin_info[u"Final_ID"] = 42
                one_pinyin_info[u"Tone"] = 'UN_KNOWN'
                one_pinyin_info[u"Tone_ID"] = 5
                pinyin_dic[item] = one_pinyin_info
        # print(pinyins)
    # pinyins = sorted(pinyin_dic.keys())
    # print(len(pinyins))

    # print(pinyins)
    return pinyin_dic


def get_initial_final_tone_sets():
    initials = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
                    'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']
    initials_ID = {'NONE': 0, 'UN_KNOWN': 24}
    ID = 1
    for initial in initials:
        initials_ID[initial] = ID
        ID += 1
    # print(initials_ID)

    initials_2 = set(['zh', 'ch', 'sh'])
    initials_1 = set(['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
                    'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w'])
    # finals = set(['i', 'u', '\"{u}', 'a', 'ia', 'ua', 'o', 'uo', 'e', 'ie',
    #               '\"{u}e', 'ai', 'uai', 'ei', 'uei', 'ao', 'iao', 'ou',
    #               'iou', 'an', 'ian', 'uan', '\"{u}an', 'en', 'in', 'uen',
    #               '\"{u}n', 'ang', 'iang', 'uang', 'eng', 'ing', 'ueng',
    #               'ong', 'iong'])
    # finals = set(['a', 'o', 'e', 'i', 'u', 'v',
    #               'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 've',
    #               'an', 'en', 'in', 'un', 'un',
    #               'ang', 'eng', 'ing', 'ong',
    #               'zhi', 'chi', 'shi', 'ri', 'zi', 'ci', 'si', 'yi', 'wu', 'yu', 'ye', 'yue'])
    finals = ['i',   'u',    'v',
              'a',   'ia',   'ua',
              'o',   'uo',
              'e',   'ie',   've',
              'ai',  'uai',
              'ei',  'uei',
              'ao',  'iao',
              'ou',  'iou',
              'an',  'ian',  'uan',  'van',
              'en',  'in',   'un',   'vn',
              'ang', 'iang', 'uang',
              'eng', 'ing',  'ueng',
              'ong', 'iong',
              'ui',  'ue',    'iu',  'ng', 'er', 'n']
    finals_ID = {'NONE': 0, 'UN_KNOWN': 42}
    ID = 1
    for final in finals:
        finals_ID[final] = ID
        ID += 1
    # print(finals_ID)

    tones_ID = {0: 'softly', 1: 'first', 2: 'second', 3: 'third', 4: 'falling', 5: 'UN_KNOWN'}
    tones = tones_ID.keys()

    return initials, initials_1, initials_2, initials_ID, finals, finals_ID, tones, tones_ID


def set_initial_final_tone(pinyin_key, values,
                           initials_1, initials_2, initials_ID,  # initials,
                           finals, finals_ID,
                           tones, tones_ID):
    if pinyin_key[0:-1] in finals:
        values[u"Initial"] = 'NONE'
        values[u"Initial_ID"] = initials_ID['NONE']
        values[u"Final"] = pinyin_key[0:-1]
        values[u"Final_ID"] = finals_ID[pinyin_key[0:-1]]
    elif pinyin_key[0:2] in initials_2:
        values[u"Initial"] = pinyin_key[0:2]
        values[u"Initial_ID"] = initials_ID[pinyin_key[0:2]]
        values[u"Final"] = pinyin_key[2:-1]
        values[u"Final_ID"] = finals_ID[pinyin_key[2:-1]]
    elif pinyin_key[0:1] in initials_1:
        values[u"Initial"] = pinyin_key[0:1]
        values[u"Initial_ID"] = initials_ID[pinyin_key[0:1]]
        values[u"Final"] = pinyin_key[1:-1]
        values[u"Final_ID"] = finals_ID[pinyin_key[1:-1]]
    else:
        if pinyin_key == 'SPECIAL_SYMBOL' or pinyin_key == 'UN_KNOWN_CHAR' or pinyin_key == 'NONE0':
            # print(pinyin_key)
            return
        else:
            print(pinyin_key)
            values[u"Initial"] = 'UN_KNOWN'
            values[u"Initial_ID"] = initials_ID['UN_KNOWN']
            values[u"Final"] = 'UN_KNOWN'
            values[u"Final_ID"] = finals_ID['UN_KNOWN']

    if int(pinyin_key[-1]) in tones:
        values[u"Tone_ID"] = int(pinyin_key[-1])
        values[u"Tone"] = tones_ID[int(pinyin_key[-1])]
    else:
        values[u"Tone_ID"] = 5
        values[u"Tone"] = tones_ID[5]
