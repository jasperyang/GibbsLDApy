#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: Utils.py
@time: 3/7/17 10:56 PM
@desc:
'''

import Constants
import re
from Strtokenizer import *

class Utils(object):

    # @tested
    def __init__(self):
        # do nothing
        print("util init")

    # @tested
    def parse_args(self,argc,argv,pmodel):
        model_status = Constants.MODEL_STATUS_UNKNOWN
        dir = ""
        model_name = ""
        dfile = ""
        alpha = -1.0
        beta = -1.0
        K = 0
        niters = 0
        savestep = 0
        twords = 0
        withrawdata = 0

        i = 0
        while i < argc:
            arg = argv[i]
            if arg == "-est":
                model_status = Constants.MODEL_STATUS_EST
            elif arg == "-estc":
                model_status = Constants.MODEL_STATUS_ESTC
            elif arg == "-inf":
                model_status = Constants.MODEL_STATUS_INF
            elif arg == "-dir":
                i += 1
                dir = argv[i]
            elif arg == "-dfile":
                i += 1
                dfile = argv[i]
            elif arg == "-model":
                i += 1
                model_name = argv[i]
            elif arg == "-alpha":
                i += 1
                alpha = float(argv[i])
            elif arg == "-beta":
                i += 1
                beta = float(argv[i])
            elif arg == "-ntopics":
                i += 1
                K = int(argv[i])
            elif arg == "-niters":
                i += 1
                niters = int(argv[i])
            elif arg == "-savestep":
                i += 1
                savestep = int(argv[i])
            elif arg == "-twords":
                i += 1
                twords = int(argv[i])
            elif arg == "-withrawdata":
                withrawdata = 1
            i += 1

        if model_status == Constants.MODEL_STATUS_EST :
            if dfile == "" :
                print("Please specify the input data file for model estimation! \n")
                return 1
            pmodel.model_status = model_status
            if K > 0 :
                pmodel.K = K
            if alpha >= 0.0 :
                pmodel.alpha = alpha
            else :
                pmodel.beta = 50.0/K
            if beta >= 0.0 :
                pmodel.beta = beta
            if niters > 0 :
                pmodel.niters = niters
            if savestep > 0 :
                pmodel.savestep = savestep
            if twords > 0 :
                pmodel.twords = twords
            pmodel.dfile = dfile
            idx = re.search('/[0-9a-zA-Z.]+$',dfile)
            if not idx :
                pmodel.dir = dir
            else :
                pmodel.dir = dfile[0:idx.start()+1]
                pmodel.dfile = dfile[idx.start()+1:]
                print("dir = ",pmodel.dir,'\n')
                print("dfile = ",pmodel.dfile,'\n')
        if model_status == Constants.MODEL_STATUS_ESTC :
            if dir == '' :
                print("Please specify model diractory!\n")
                return 1
            if model_name == '' :
                print("Please specify model name upon that you want to continue estimating!\n")
                return 1
            pmodel.model_status = model_status
            if dir[len(dir)-1] != '/' :
                dir += '/'
            pmodel.dir = dir
            pmodel.model_name = model_name
            if niters > 0 :
                pmodel.niters = niters
            if savestep > 0 :
                pmodel.savestep = savestep
            if twords > 0 :
                pmodel.twords = twords
            # read <model>.others file to assign values for ntopics, alpha, beta, etc.
            if self.read_and_parse(pmodel.dir + pmodel.model_name + pmodel.others_suffix,pmodel) :
                return 1
        if model_status == Constants.MODEL_STATUS_INF :
            if dir == '' :
                print("Please specify model diractory!\n")
                return 1
            if model_name == '' :
                print("Please specify model name for inference!\n")
                return 1
            if dfile == '' :
                print("Please specify the new data file for inference!\n")
                return 1
            pmodel.model_status = model_status
            if dir[len(dir) - 1] != '/':
                dir += '/'
            pmodel.dir = dir
            pmodel.model_name = model_name
            pmodel.dfile = dfile
            if niters > 0:
                pmodel.niters = niters
            else :
                pmodel.niters = 20
            if twords > 0:
                pmodel.twords = twords
            if withrawdata > 0 :
                pmodel.withrawstrs = withrawdata
            # read <model>.others file to assign values for ntopics, alpha, beta, etc.
            if self.read_and_parse(pmodel.dir + pmodel.model_name + pmodel.others_suffix, pmodel):
                return 1
        if model_status == Constants.MODEL_STATUS_UNKNOWN :
            print("Please specify the task you would list to perform (-est/-estc/inf)!\n")
            return 1

        return 0

    # @tested
    def read_and_parse(self,filename,pmodel):
        # open file <model>.others to read:
        # alpha=?
        # beta=?
        # ntopics=?
        # ndocs=?
        # nwords=?
        # citer=?  # current iteration (when the model was saved)
        fin = open(filename)
        if not fin :
            print("Cannot open file ",filename," \n")
            return 1
        line = fin.readline()
        while line :
            strtok = Strtokenizer(line,'= \t\r\n')
            count = strtok.count_tokens()
            if count != 2 :
                continue
            optstr = strtok.token(0)
            optval = strtok.token(1)
            if optstr == 'alpha' :
                pmodel.alpha = float(optval)
            elif optstr == 'beta' :
                pmodel.beta = float(optval)
            elif optstr == 'ntopics' :
                pmodel.K = int(optval)
            elif optstr == 'ndocs' :
                pmodel.M = int(optval)
            elif optstr == 'nwords' :
                pmodel.V = int(optval)
            elif optstr == 'liter' :
                pmodel.liter = int(optval)
            line = fin.readline()
        fin.close()
        return 0

    # @tested
    def generate_model_name(self,iter):
        model_name = 'model-'
        buff = ''
        if 0 <= iter and iter < 10 :
            buff += '0000' + str(iter)
        elif 10 <= iter and iter < 100 :
            buff += '000' + str(iter)
        elif 100 <= iter and iter < 1000:
            buff += '00' + str(iter)
        elif 1000 <= iter and iter < 10000:
            buff += '0' + str(iter)
        else :
            buff += str(iter)

        if iter >= 0 :
            model_name += buff
        else :
            model_name += 'final'
        return model_name

    # @tested
    # 冒泡排序...-_-
    def sort(self,probs,words):
        for i in range(len(probs)-1) :
            for j in range(i+1,len(probs)) :
                if probs[i] < probs[j] :
                    tempprob = probs[i]
                    tempword = words[i]
                    probs[i] = probs[j]
                    words[i] = words[j]
                    probs[j] = tempprob
                    words[j] = tempword
        return 0

    # @tested
    # 归并排序...  本来vect用的是 vector<pair<int,double>> 的数据结构,在python里面我就用 list[dict,dict,...]这种方式来代替了
    def quicksort(self,vect,left,right):
        l_hold = left
        r_hold = right
        pivotidx = left
        pivot = vect[pivotidx]      # pivot 是 dict

        while left < right :
            while list(vect[right].values())[0] <= list(pivot.values())[0] and left < right :       # 这里有个强制转换成list的trick,本来是view,这是个更加dynamic的数据结构
                right -= 1
            if left != right :
                vect[left] = vect[right]
                left += 1
            while list(vect[left].values())[0] >= list(pivot.values())[0] and left < right :
                left += 1
            if left != right :
                vect[right] = vect[left]
                right -= 1
        vect[left] = pivot
        pivotidx = left
        left = l_hold
        right = r_hold

        if left < pivotidx :
            self.quicksort(vect,left,pivotidx-1)
        if right > pivotidx :
            self.quicksort(vect,pivotidx+1,right)
        return 0