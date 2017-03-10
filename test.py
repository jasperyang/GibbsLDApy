#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: test.py
@time: 3/6/17 8:41 PM
@desc: This is for testing!!! all the functions
'''

from DataSet import *
from Document import *
from Strtokenizer import *
from Utils import *
from Model import *

'''test of strtokenizer'''
# line = 'fasdf asd f dsaf ds af dsaf sdaf dsa fd saf s '
# strtok = Strtokenizer(line,' \r\t\n')
# for i in range(strtok.count_tokens()) :
#     print(str(i) + ":" + strtok.token(i) + '\n')
# print(strtok.next_token())

'''test of document'''
# dd = Document()
# line = 'adf'
# do = Document(dd,line)
# print(do.rawstr)            # Document这个类可用

'''test of dataset'''  # wordmap.txt 中不能出现两个空格连在一起,最后一行要加上换行
wordmapfile = 'test_data/wordmap.txt'
da = DataSet(2)

# pword2id = {'nihao':1,'sa':2,'hahaha':3}
# da.write_wordmap(wordmapfile,pword2id)

# pword2id = {}
# da.read_wordmap1(wordmapfile,pword2id)
# for key,value in pword2id.items() :
#     print(key + str(value))

# pid2word = {}
# da.read_wordmap2(wordmapfile,pid2word)
# for key,value in pid2word.items() :
#     print(key + str(value))

# dfile = 'test_data/dfile'
# da.read_trndata(dfile,wordmapfile)
# for doc in da.docs :
#     print(doc.words)
# da.read_newdata('test_data/newdfile',wordmapfile)
# for doc in da.docs :
#     print(doc.words)
# da.read_newdata_withrawstrs('test_data/new2dfile',wordmapfile)
# for doc in da.docs :
#     print(doc.words)


'''test of Util'''
# argv = ['-estc', '-alpha', '0.5', '-beta', '0.1', '-ntopics', '100', '-niters',
#         '1000', '-savestep', '100', '-twords', '20', '-dfile', 'models/casestudy/trndocs.dat', '-dir', 'test_data',
#         '-model', 'model-01800']
# pmodel = Model()
# u = Utils()
# u.parse_args(len(argv), argv, pmodel)
# print(u.generate_model_name(80))

# probs = [2.4,54.23,1.4]
# words = [0,1,2]
# u.sort(probs,words)
# print(probs)
# print(words)

# vect = [{0:2.4},{1:54.23},{2:1.4}]
# u.quicksort(vect,0,2)
# print(vect)

'''test of model'''

# # 不包括需要load_model的
argv = ['-est', '-alpha', '0.5', '-beta', '0.1', '-ntopics', '10', '-niters',
        '1000', '-savestep', '100', '-twords', '20', '-dfile', 'dfile', '-dir', 'test_data/',
        '-model', 'testmodel']
pmodel = Model()
pmodel.init(len(argv),argv)                     # 测试 init 包括 init_est
# print("nw:\n")
# print(pmodel.nw)
# print("nd:\n")
# print(pmodel.nd)
# print("nwsum:\n")
# print(pmodel.nwsum)
# print("ndsum:\n")
# print(pmodel.ndsum)
# print("z:\n")
# print(pmodel.z)

# pmodel.load_model('testmodel')
# print(pmodel.z)

# pmodel.save_model_tassign('test_data/testmodel.tassign')
# pmodel.save_model_theta('test_data/testmodel.theta')
# pmodel.save_model_phi('test_data/testmodel.phi')
# pmodel.save_model_twords('test_data/testmodel.twords')
# pmodel.save_model_others('test_data/testmodel.others')
# pmodel.save_model('testmodel')


# 包括需要load_model的   init_estc,init_inf
# argv = ['-inf', '-alpha', '0.5', '-beta', '0.1', '-ntopics', '10', '-niters',
#         '1000', '-savestep', '100', '-twords', '20', '-dfile', 'dfile', '-dir', 'test_data/',
#         '-model', 'testmodel']
# pmodel = Model()
# pmodel.init(len(argv),argv)
# pmodel.save_inf_model('test_inf_model')
# pmodel.save_inf_model_tassign('test_data/test_inf_model.tassign')
# pmodel.save_inf_model_newtheta('test_data/test_inf_model.theta')
# pmodel.save_inf_model_newphi('test_data/test_inf_model.phi')
# pmodel.save_inf_model_twords('test_data/test_inf_model.twords')


# pmodel.estimate()
# print(pmodel.z)