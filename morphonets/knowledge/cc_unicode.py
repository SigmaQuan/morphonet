# -*- coding: utf-8 -*-
"""
Get unicode of Chinese characters come from Universal Standard Chinese Character
List which released by Chinese State Council at 2013.
"""
import cc_level_1
import cc_level_2


def get_unicode_string(character):
    character_string = repr(character)
    removed_string = "u'\\u"
    if len(character_string) > 9:  # and len(character) == 2
        removed_string = "u'\\U000"
        # print(character)
        # print(character_string)
        # print(removed_string)
    unicode_string = character_string.replace(removed_string, "")
    unicode_string = unicode_string.replace("'", "")
    unicode_string = unicode_string.upper()
    # if len(unicode_string) > 4:
    #     print(unicode_string)
    return unicode_string


def get():
    unicode_dic = {}

    level_1_3500_characters = cc_level_1.get_characters()
    for character in level_1_3500_characters:
        unicode_dic[character] = get_unicode_string(character)

    level_2_3000_characters = cc_level_2.get_characters()
    # print(level_2_3000_characters[689])
    # print(level_2_3000_characters[2488])
    for character in level_2_3000_characters:
        unicode_dic[character] = get_unicode_string(character)

    return unicode_dic
