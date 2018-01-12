# -*- coding: utf-8 -*-
import requests

# zdic = "http://www.zdic.net/z/17/js/5973.htm"
# zdic = "http://www.zdic.net/search/?q=𥻗"
httpcn = "http://tool.httpcn.com/zi/"


def get_URL(character):
    url = ""
    try:
        kv = {'q': character}
        r = requests.get(httpcn, params=kv)
        r.raise_for_status()
        url = r.request.url
    except:
        print("Download error: " + httpcn)
    return url


def get_page_text(character):
    page_text = u""
    try:
        kv = {'q': character}
        r = requests.get(httpcn, params=kv)
        print(character)
        print(r.request.url)
        r.raise_for_status()
        # print(r.status_code)
        #  print(r.encoding)
        #  print(r.apparent_encoding)
        r.encoding = r.apparent_encoding
        # print("page text length: %d" % len(r.text))
        # print("page text: \n", r.text)
        print(r.text)
        page_text = r.text
    except:
        print(u"Download error: " + repr(character))

    return page_text