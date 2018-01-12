# -*- coding: utf-8 -*-
"""
Get Pinyin and Radical information of Chinese characters.
"""
from bs4 import BeautifulSoup
import os
import json

import cc_level_1 as level_1_3500_2013
import cc_level_2 as level_2_3000_2013
from cc_unicode import get_unicode_string
from crawler.zdic_net import get_URL, get_page_text

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
current_py_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_py_file))))
from configs.globals import PROJECT_FOLDER
DATA_FOLDER = PROJECT_FOLDER + "/data/knowledge/"


def get_morphology():
    morphology_dict = {}

    return morphology_dict


def get_morphology(chinese_character):
    morphology_dict = get_morphology()
    return morphology_dict[chinese_character]


def get_phonology():
    phonology_dict = {}

    return phonology_dict


def get_phonology(chinese_character):
    phonology_dict = get_phonology()
    return phonology_dict[chinese_character]


def get_pinyin_and_radical():
    dump_file = DATA_FOLDER + 'cc_dictionary_filled.txt'
    if os.path.exists(dump_file):
        return json.load(open(dump_file, 'r'))
    else:
        pass

    dic = initial_dictionary()
    print("Total Chinese characters: %d" % len(dic))
    for (character, info) in dic.items():
        print "character: " + character + "; unicode: " + \
              info[u"Unicode"] + "; URL: " + info[u"URL"]  #  + \
              # "; Page text: " + info[u"PageText"]

        soup = BeautifulSoup(dic[character][u"PageText"], "html.parser")
        dic[character][u"PinYin"] = parse_pinyin(soup)
        dic[character][u"Radical"] = parse_radical(soup)
        dic[character][u"Structure"] = parse_structure(soup)
        dic[character][u"Stroke"] = parse_stroke(soup)
        del dic[character][u"PageText"]
        del dic[character][u"Component"]

    json.dump(dic, open('cc_dictionary_filled.txt', 'w'))

    return dic


def initial_dictionary():
    dump_file = DATA_FOLDER + 'cc_dictionary_new.txt'
    if os.path.exists(dump_file):
        return json.load(open(dump_file, 'r'))
    else:
        pass

    dic = {}
    characters_level_1 = level_1_3500_2013.get_characters()
    characters_level_2 = level_2_3000_2013.get_characters()
    characters = characters_level_1 + characters_level_2
    print(len(characters))
    print(characters[3500 + 502])
    print(characters[3500 + 503])
    print(characters[3500 + 504])
    print(characters[3500 + 688])
    print(characters[3500 + 689])
    print(characters[3500 + 2489-1])
    print(characters[3500 + 2487])

    for character in characters:
        one_character_info = {}
        one_character_info[u"Unicode"] = get_unicode_string(character)
        one_character_info[u"PinYin"] = {}
        one_character_info[u"Component"] = {}
        one_character_info[u"URL"] = get_URL(character)
        one_character_info[u"PageText"] = get_page_text(character)
        dic[character] = one_character_info

    json.dump(dic, open(dump_file, 'w'))

    return dic


def parse_pinyin(soup):
    html_pinyin = soup.find_all('a')
    pinyins = []
    for py in html_pinyin:
        text = py.get('href')
        if len(text) > 12 and text[0:12] == u"/z/pyjs/?py=":
            # print(text[12:])
            pinyins.append(text[12:])

    return pinyins


def parse_radical(soup):
    html_radical = soup.find_all('a')
    radical = {}
    radical[u"Head"] = u""
    radical[u"HeadLink"] = u""
    radical[u"Other"] = u""
    radical[u"OtherLink"] = u""
    head_flag = 0
    for py in html_radical:
        link = py.get('href')
        if len(link) > 12 and link[0:12] == u"/z/jbs/?jbs=" and head_flag == 0:
            # print(link[12:])
            radical[u"Head"] = py.text
            radical[u"HeadLink"] = link
            head_flag = 1
            continue

        if len(link) > 16 and link[0:12] == u"/z/jbs/?jbs=" and link[12:].find(u'jbh='):
            # print(link[12:])
            radical[u"Other"] = py.text
            radical[u"OtherLink"] = link
            break
    # print(radical)

    return radical


def parse_structure(soup):
    html_structure = soup.find_all('a')
    structure = {}
    structure[u"Type"] = u""
    structure[u"Content"] = u""
    structure[u"Link"] = u""
    for py in html_structure:
        link = py.get('href')
        if len(link) > 14 and link[0:14] == u"/z/zxjs/?zxjg=":
            # print(link[14:])
            structure[u"Type"] = py.text
            structure[u"Link"] = link
            break

    html_structure_content = soup.find_all('td', 'z_i_t2')
    if len(html_structure_content) == 2:
        content = html_structure_content[1]
        if len(content.contents) == 2:
            structure[u"Content"] = content.contents[1]

    print(structure[u"Type"])
    print(structure[u"Content"])
    print(structure[u"Link"])

    return structure


def parse_stroke(soup):
    html_stroke = soup.find_all('span')
    stroke = {}
    stroke[u"Number"] = u""
    stroke[u"Write"] = u""
    for py in html_stroke:
        id = py.get('id')
        if id == u"z_i_t2_bis":
            # print(link[12:])
            stroke[u"Number"] = py.get('title')
            stroke[u"Write"] = py.text
            break
    # print(stroke)

    return stroke
