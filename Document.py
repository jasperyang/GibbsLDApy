#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: Document.py
@time: 3/6/17 7:27 PM
@desc:
'''

class Document(object):
    words = None  # 每个词对应的id
    rawstr = ''  # 无修改的原文本
    length = 0  # 文档的单词个数


    # 难用的构造函数
    def __init__(self,*argv):
        if len(argv) == 0 :
            self.words = None
            self.rawstr = ''
            self.length = 0
        elif len(argv) == 1 :
            if type(argv[0]) == int :       #length
                self.length = argv[0]
                self.rawstr = ''
                self.words = []  # words 是 list 类型的方便操作
            else :      # doc 这个doc是个list<int>
                self.length = len(argv[0])
                self.rawstr = ''
                self.words = []
                for i in range(self.length):
                    self.words.append(argv[0][i])
        elif len(argv) == 2 :
            if type(argv[0]) == int:  # length,words
                self.length = argv[0]
                self.rawstr = ''
                self.words = argv[1]
            else :  # doc,rawstr
                self.length = len(argv[0])
                self.rawstr = argv[1]
                self.words = []
                for i in range(self.length):
                    self.words.append(argv[0][i])
        elif len(argv) == 3 :
            self.length = argv[0]
            self.rawstr = argv[1]
            self.words = argv[2]
        else :
            print("invalid init")

    def __del__(self):
        return