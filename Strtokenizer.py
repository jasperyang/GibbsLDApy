#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: Strtokenizer.py
@time: 3/6/17 8:20 PM
@desc:
'''

import re

class Strtokenizer(object):
    tokens = []
    idx = 0

    def __init__(self):
        self.tokens = []
        self.idx = 0

    def __init__(self,str,seperators = ''):
        self.tokens = []       # 事先清空,和c++中的有区别
        self.parse(str,seperators)

    def find_first_not_of(self,seperators,str):
        pattern = '['+seperators+']'
        if re.search(pattern,str) :
            if re.search(pattern,str).span() != (0,1) :
                start = 0
            else :
                index = 2
                string = str[index:]
                if re.search(pattern, string):
                    while re.search(pattern,string).span() == (0,1) :
                        index += 2
                        string = str[index:]
                start = index
        else:
            start = 0
        return start

    def parse(self,str,seperators):
        n = len(str)
        start = self.find_first_not_of(seperators,str)
        if start != 0 :
            str = str[start-1:]
        start = 0
        pattern = '['+seperators+']'
        while start >= 0 and start < n :
            if re.search(pattern, str):
                stop = int(re.search(pattern,str).span()[0])
            else:
                stop = -1
            if stop < 0 or stop > n :       # 这里的是or而不是and,找了快两个小时...
                stop = n
            self.tokens.append(str[start:stop])
            str = str[stop+1:]
            n = len(str)
            if n == 0 :
                break
            start = self.find_first_not_of(seperators,str)


        self.start_scan()

    def count_tokens(self):
        return len(self.tokens)

    def start_scan(self):
        self.idx = 0

    def next_token(self):
        if self.idx >= 0 and self.idx < len(self.tokens) :
            self.idx += 1
            return self.tokens[self.idx-1]
        else :
            return  ''

    def token(self,i):
        if i >= 0 and i < len(self.tokens) :
            return self.tokens[i]
        else :
            return ''