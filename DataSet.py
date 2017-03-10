#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: DataSet.py
@time: 3/6/17 7:27 PM
@desc:
'''

import Constants
from Document import *
from Strtokenizer import *

class  DataSet(object):

    def __init__(self,*argv):
        if len(argv) == 0:
            self.docs = []
            self._docs = []  # used only for inference
            self._id2id = {}  # also used only for inference
            self.M = ''  # number of documents
            self.V = ''  # number of words
        elif len(argv) == 1:
            self.docs = []
            self._docs = []  # used only for inference
            self._id2id = {}  # also used only for inference
            self.M = argv[0]  # number of documents
            self.V = ''  # number of words
        else:
            print("invalid init")

    def __del__(self):
        return

    def add_doc(self,doc,idx):
        if 0 <= idx and idx < self.M :
            if len(self.docs) > idx :
                self.docs[idx] = doc
            else :
                self.docs.append(doc)

    def _add_doc(self,_doc,idx):
        if 0 <= idx and idx < self.M :
            if len(self._docs) > idx:
                self._docs[idx] = _doc
            else :
                self._docs.append(_doc)

    # @tested
    def write_wordmap(self,wordmapfile,pword2id):
        fout = open(wordmapfile,'w')
        if fout == None :
            print("Cannot open file ", wordmapfile, " to write!\n")
            return 1
        fout.write(str(len(pword2id)) + "\n")
        for key,value in pword2id.items():
            fout.write(key + ' ' + str(value) +'\n')
        fout.close()
        return 0

    # @tested
    def read_wordmap1(self,wordmapfile,pword2id):
        pword2id.clear()        # 清空字典
        fin = open(wordmapfile)
        if fin == None :
            print("Cannot open file ",wordmapfile," to read!\n")
            return 1
        line = fin.readline()
        nwords = int(line)
        for i in range(nwords):
            line = fin.readline()
            strtok = Strtokenizer(line,' \t\r\n')
            if strtok.count_tokens() != 2 :
                continue
            pword2id[strtok.token(0)] = int(strtok.token(1))
        fin.close()
        return 0

    # @tested
    def read_wordmap2(self,wordmapfile,pid2word):
        pid2word.clear()
        fin = open(wordmapfile)
        if fin == None :
            print("Cannot open file ", wordmapfile, " to read!\n")
            return 1
        line = fin.readline()
        nword = int(line)
        for i in range(nword) :
            line = fin.readline()
            strtok = Strtokenizer(line,' \t\r\n')
            if strtok.count_tokens() != 2 :
                continue
            pid2word[strtok.token(1)] = strtok.token(0)
        fin.close()
        return 0

    # @tested
    # 读取训练数据  生成wordmap.txt 里面对应的是单词和单词id
    # 每个doc里面存储的是里面所有的word对应的id
    def read_trndata(self,dfile,wordmapfile):
        word2id = {}
        fin = open(dfile)
        if fin == None :
            print("Cannot open file ", wordmapfile, " to read!\n")
            return 1
        line = fin.readline()
        self.M = int(line)
        if self.M <= 0 :
            print("No document available!\n")
            return 1
        self.V = 0
        for i in range(self.M) :
            line = fin.readline()
            strtok = Strtokenizer(line,' \t\r\n')    # 按照一行一行来划分(其实一行就是一篇文档)
            length = strtok.count_tokens()
            if length <= 0 :
                print("Invalid (empty) document!\n")
                self.M = self.V =0
                return 1
            pdoc = Document(length)
            for j in range(length) :
                found = False
                for key in word2id.keys() :         # 这里应该是去和key匹配而不是value,检查了好久!!!
                    if strtok.token(j) == key :
                        found = True
                        break
                if not found : # 没找到
                    pdoc.words.append(len(word2id))     # 这篇文档里有哪个编号的单词,加进去
                    word2id[strtok.token(j)] = len(word2id)    # 给每个单词标号
                else :
                    pdoc.words.append(word2id[strtok.token(j)])
            self.add_doc(pdoc,i)
        fin.close()
        if self.write_wordmap(wordmapfile,word2id) :
            return 1
        self.V = len(word2id)
        return 0

    # @tested
    # 读取预测数据
    # id2_id : id是原文档中单词对应id , _id是新的文档中出现的原文档的词的编号(也是从0开始的,这个很难理解)
    # _id2id : 以后可以用到
    # 一个神奇的理解了很久的地方,就是新的文档中的出现的新词是不管的
    def read_newdata(self,dfile,wordmapfile):
        word2id = {}
        id2_id = {}
        self.read_wordmap1(wordmapfile,word2id)   # 从wordmapfile中读入word2id
        if len(word2id) <= 0 :
            print("No word map available!\n")
            return 1
        fin = open(dfile)
        if fin == None :
            print("Cannot open file ",dfile," to read!\n")
            return 1
        line = fin.readline()
        # get the number of new documents
        self.M = int(line)
        if self.M <= 0 :
            print("No document available!\n")
            return 1
        self.V = 0

        for i in range(self.M) :
            line = fin.readline()
            strtok = Strtokenizer(line,' \t\r\n')
            length = strtok.count_tokens()
            doc = []
            _doc = []
            for j in range(length) :
                found = False
                for key in word2id.keys():
                    if strtok.token(j) == key:
                        found = True
                        break
                if found:   # 找到 key
                    found2 = False
                    for value in id2_id.values() :
                        if value == word2id[strtok.token(j)] :
                            found2 = True
                            break
                    if not found2 : # 没找到 value
                        _id = len(id2_id)
                        id2_id[word2id[strtok.token(j)]] = _id
                        self._id2id[_id] = word2id[strtok.token(j)]
                    else :  # 找到
                        _id = word2id[strtok.token(j)]
                    doc.append(word2id[strtok.token(j)])
                    _doc.append(_id)
                else:           # 没找到
                    tmp = ''
                    # 没决定要做什么

            pdoc = Document(doc)
            _pdoc = Document(_doc)
            self.add_doc(pdoc,i)
            self._add_doc(_pdoc,i)

        fin.close()
        self.V = len(id2_id)
        return 0


    # @tested
    def read_newdata_withrawstrs(self,dfile,wordmapfile):
        word2id = {}
        id2_id = {}
        self.read_wordmap1(wordmapfile,word2id)   # 从wordmapfile中读入word2id
        if len(word2id) <= 0 :
            print("No word map available!\n")
            return 1
        fin = open(dfile)
        if fin == None :
            print("Cannot open file ",dfile," to read!\n")
            return 1
        line = fin.readline()
        self.M = int(line)
        if self.M <= 0 :
            print("No document available!\n")
            return 1
        self.V = 0

        for i in range(self.M) :
            line = fin.readline()
            strtok = Strtokenizer(line,' \t\r\n')
            length = strtok.count_tokens()
            doc = []
            _doc = []
            for j in range(length) :
                found = False
                for key in word2id.keys():
                    if strtok.token(j) == key:
                        found = True
                        break
                if found:   # 找到 key
                    found2 = False
                    for value in id2_id.values() :
                        if value == word2id[strtok.token(j)] :
                            found2 = True
                            break
                    if not found2 : # 没找到 value
                        _id = len(id2_id)
                        id2_id[word2id[strtok.token(j)]] = _id
                        self._id2id[_id] = word2id[strtok.token(j)]
                    else :  # 找到
                        _id = word2id[strtok.token(j)]
                    doc.append(word2id[strtok.token(j)])
                    _doc.append(_id)
                else:      # 没找到
                    tmp = ''
                    # 没决定要做什么
            pdoc = Document(doc,line)
            _pdoc = Document(_doc,line)

            self.add_doc(pdoc,i)
            self._add_doc(_pdoc,i)
        fin.close()
        self.V = len(id2_id)
        return 0