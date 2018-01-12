# -*- coding: utf-8 -*-
import os
import sys
import time
import pytest

from configs.globals import PROJECT_FOLDER
from morphonets.knowledge.crawler import zdic_net
from morphonets.knowledge.crawler import httpcn


FOLDER = PROJECT_FOLDER + \
         time.strftime('/logs/knowledge/crawler/%Y-%m-%d-%H-%M-%S/')


def test_zdic_net(folder=FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/corpus_statistics.log' % folder
    sys.stdout = open(log_file, 'a')

    # keyword = "𥻗"
    keyword = "𠳐"
    # zdic = "http://www.zdic.net/search/?q=𥻗"
    zdic = "http://www.zdic.net/search/"

    print(keyword)
    print(zdic_net.get_URL(keyword))
    print((zdic_net.get_page_text(keyword)).encode('utf-8'))

    sys.stdout.close_board()
    sys.stdout = sys_stdout


def test_httpcn_net(folder=FOLDER):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("Create folder: %s" % folder)
    sys_stdout = sys.stdout
    log_file = '%s/corpus_statistics.log' % folder
    sys.stdout = open(log_file, 'a')

    # # keyword = "𥻗"
    # keyword = "𠳐"
    # # zdic = "http://www.zdic.net/search/?q=𥻗"
    # zdic = "http://www.zdic.net/search/"
    #
    # print(keyword)
    # print(httpcn.get_URL(keyword))
    # print((httpcn.get_page_text(keyword)).encode('utf-8'))

    sys.stdout.close_board()
    sys.stdout = sys_stdout

if __name__ == '__main__':
    pytest.main([__file__])
