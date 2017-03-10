#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: Model.py
@time: 3/7/17 11:00 PM
@desc:    一些变量的解释
          1.一个训练数据的dataset:ptrndata
          2.word-topic的矩阵:nw
          3.某篇文档对应的各种topic的数量的矩阵:nd
          4.某个topic中总的单词数的矩阵:nwsum
          5.某个文档中单词数的矩阵:ndsum
          6.每篇文档中每个单词对应的topic的概率矩阵:z
'''

import Constants
from Utils import *
from DataSet import *
import random
import numpy as np


'''后来发现要声明一个二维数组很简单,b = [[0]*10]*10 ...'''

class Model(object):
    wordmapfile = ''  # file that contains word map [string -> integer id]
    trainlogfile = ''  # training log file
    tassign_suffix = ''  # suffix for topic assignment file
    theta_suffix = ''  # suffix for theta file
    phi_suffix = ''  # suffix for phi file
    others_suffix = ''  # suffix for file containing other parameters
    twords_suffix = ''  # suffix for file containing words-per-topics

    dir = ''  # model directory
    dfile = ''  # data file
    model_name = ''  # model name
    model_status = None  # model status:
    #               MODEL_STATUS_UNKNOWN: unknown status
    #               MODEL_STATUS_EST: estimating from scratch
    #               MODEL_STATUS_ESTC: continue to estimate the model from a previous one
    #               MODEL_STATUS_INF: do inference
    ptrndata = []  # list of training dataset object
    pnewdata = []  # list of new dataset object

    id2word = {}  # word map [int => string]

    # --- model parameters and variables ---
    M = None  # dataset size (i.e., number of docs)
    V = None  # vocabulary size
    K = None  # number of topics
    alpha = None  # LDA hyperparameters
    beta = None
    niters = None  # number of Gibbs sampling iterations
    liter = None  # the iteration at which the model was saved
    savestep = None  # saving period
    twords = None  # print out top words per each topic
    withrawstrs = None

    p = []  # temp variable for sampling
    z = []  # topic assignments for words, size M x doc.size()
    nw = []  # cwt[i][j]: number of instances of word/term i assigned to topic j, size V x K
    nd = []  # na[i][j]: number of words in document i assigned to topic j, size M x K
    nwsum = []  # nwsum[j]: total number of words assigned to topic j, size K
    ndsum = []  # nasum[i]: total number of words in document i, size M
    theta = []  # theta: document-topic distributions, size M x K
    phi = []  # phi: topic-word distributions, size K x V

    # for inference only
    inf_liter = None
    newM = None
    newV = None
    newz = []
    newnw = []
    newnd = []
    newnwsum = []
    newndsum = []
    newtheta = []
    newphi = []

    # --------------------------------------

    # @tested
    def __init__(self):
        self.setdefault_value()

    # @tested
    def setdefault_value(self):
        self.wordmapfile = "wordmap.txt"
        self.trainlogfile = "trainlog.txt"
        self.tassign_suffix = ".tassign"
        self.theta_suffix = ".theta"
        self.phi_suffix = ".phi"
        self.others_suffix = ".others"
        self.twords_suffix = ".twords"

        self.dir = "./"
        self.dfile = "trndocs.dat"
        self.model_name = "model-final"
        self.model_status = Constants.MODEL_STATUS_UNKNOWN

        self.ptrndata = None
        self.pnewdata = None

        self.M = 0
        self.V = 0
        self.K = 100
        self.alpha = 50.0 / self.K
        self.beta = 0.1
        self.niters = 2000
        self.liter = 0
        self.savestep = 200
        self.twords = 0
        self.withrawstrs = 0

        self.p = None
        self.z = None
        self.nw = None
        self.nd = None
        self.nwsum = None
        self.ndsum = None
        self.theta = None
        self.phi = None

        self.newM = 0
        self.newV = 0
        self.newz = None
        self.newnw = None
        self.newnd = None
        self.newnwsum = None
        self.newndsum = None
        self.newtheta = None
        self.newphi = None

    # @tested
    def parse_args(self, argc, argv):
        u = Utils()
        return u.parse_args(argc, argv, self)

    # @tested
    def init(self, argc, argv):
        # call parse_args
        if self.parse_args(argc, argv):
            return 1
        if self.model_status == Constants.MODEL_STATUS_EST:
            # estimating the model from scratch (从头开始分析模型)
            if self.init_est():
                return 1
        elif self.model_status == Constants.MODEL_STATUS_ESTC:
            if self.init_estc():
                return 1
        elif self.model_status == Constants.MODEL_STATUS_INF:
            if self.init_inf():
                return 1
        return 0

    # @tested
    # 从头开始分析,初始化模型
    def init_est(self):
        self.p = []  # double[K]
        for k in range(self.K):
            self.p.append(0)

        # + read training data
        ptrndata = DataSet()
        if ptrndata.read_trndata(self.dir + self.dfile, self.dir + self.wordmapfile):
            print("Fail to read training data!\n")
            return 1

        # + assign values for variables
        self.M = ptrndata.M
        self.V = ptrndata.V
        # K: from command line or default value
        # alpha, beta: from command line or default values
        # niters, savestep: from command line or default values

        self.nw = []  # int[V]
        for w in range(self.V):
            nw_row = []  # int[K]
            for k in range(self.K):
                nw_row.append(0)
            self.nw.append(nw_row)

        self.nd = []  # int[M]
        for w in range(self.M):
            nd_row = []  # int[K]
            for k in range(self.K):
                nd_row.append(0)
            self.nd.append(nd_row)

        self.nwsum = []  # int[K]
        for k in range(self.K):
            self.nwsum.append(0)

        self.ndsum = []  # int[M]
        for k in range(self.M):
            self.ndsum.append(0)

        self.z = []  # int[M]
        for m in range(self.M):
            self.z.append([])


        for m in range(self.M):
            N = ptrndata.docs[m].length   # 该篇文档的单词数

            for n in range(N):  # 初始化z[M][N]
                self.z[m].append(0)

            # initialize for z
            for n in range(N):  # 遍历每个单词
                topic = random.randint(0, self.K-1)
                self.z[m][n] = topic                # 给单词随机赋予主题

                # number of instance of word i assigned to topic j 单词对应文档 V * K
                self.nw[ptrndata.docs[m].words[n]][topic] += 1
                # number of words in document i assigned to topic j  M * K
                self.nd[m][topic] += 1
                # total number of words assigned to topic j
                self.nwsum[topic] += 1
            # total number of words in document i
            self.ndsum[m] = N

        self.theta = []  # double[m]
        for m in range(self.M):
            theta_row = []  # double[K]
            for k in range(self.K):
                theta_row.append(0)
            self.theta.append(theta_row)

        self.phi = []  # double[K]
        for k in range(self.K):
            phi_row = []  # double[V]
            for v in range(self.V):
                phi_row.append(0)
            self.phi.append(phi_row)

        self.ptrndata = ptrndata
        return 0

    # @testing
    def init_estc(self):
        # estimating the model from a previously estimated one
        self.p = []  # double[K]
        for k in range(self.K):
            self.p.append(0)

        # load model , i.e., read z and ptrndata
        if self.load_model(self.model_name):
            print("Fail to load word-topic assignment file of the model!\n")
            return 1

        self.nw = []  # int[V]
        for w in range(self.V):
            nw_row = []  # int[K]
            for k in range(self.K):
                nw_row.append(0)
            self.nw.append(nw_row)

        self.nd = []  # int[M]
        for w in range(self.M):
            nd_row = []  # int[K]
            for k in range(self.K):
                nd_row.append(0)
            self.nd.append(nd_row)

        self.nwsum = []  # int[K]
        for k in range(self.K):
            self.nwsum.append(0)

        self.ndsum = []  # int[M]
        for k in range(self.M):
            self.ndsum.append(0)

        self.z = []  # int[M]
        for m in range(self.M):
            self.z.append([])

        for m in range(self.ptrndata.M):
            N = self.ptrndata.docs[m].length

            for n in range(N):  # 初始化z[M][N]
                self.z[m].append(0)

            # assign values for nw, nd, nwsum, and ndsum
            for n in range(N):
                w = self.ptrndata.docs[m].words[n]
                topic = self.z[m][n]
                # number of instance of word i assigned to topic j
                self.nw[w][topic] += 1
                # number of words in document i assigned to topic j
                self.nd[m][topic] += 1
                # total number of words assigned to topic j
                self.nwsum[topic] += 1
            # total number of words in document i
            self.ndsum[m] = N

        self.theta = []  # double[m]
        for m in range(self.M):
            theta_row = []  # double[K]
            for k in range(self.K):
                theta_row.append(0)
            self.theta.append(theta_row)

        self.phi = []  # double[K]
        for k in range(self.K):
            phi_row = []  # double[V]
            for v in range(self.V):
                phi_row.append(0)
            self.phi.append(phi_row)

        return 0

    # @tested
    def init_inf(self):
        # estimating the model from a previously estimated one
        self.p = []  # double[K]
        for k in range(self.K):
            self.p.append(0)

        # load model , i.e., read z and ptrndata
        if self.load_model(self.model_name):
            print("Fail to load word-topic assignment file of the model!\n")
            return 1

        self.nw = []  # int[V]
        for w in range(self.V):
            nw_row = []  # int[K]
            for k in range(self.K):
                nw_row.append(0)
            self.nw.append(nw_row)

        self.nd = []  # int[M]
        for w in range(self.M):
            nd_row = []  # int[K]
            for k in range(self.K):
                nd_row.append(0)
            self.nd.append(nd_row)

        self.nwsum = []  # int[K]
        for k in range(self.K):
            self.nwsum.append(0)

        self.ndsum = []  # int[M]
        for k in range(self.M):
            self.ndsum.append(0)

        self.z = []  # int[M]
        for m in range(self.M):
            self.z.append([])

        for m in range(self.ptrndata.M):
            N = self.ptrndata.docs[m].length

            for n in range(N):  # 初始化z[M][N]
                self.z[m].append(0)

            # assign values for nw, nd, nwsum, and ndsum
            for n in range(N):
                w = self.ptrndata.docs[m].words[n]
                topic = self.z[m][n]

                # number of instance of word i assigned to topic j
                self.nw[w][topic] += 1
                # number of words in document i assigned to topic j
                self.nd[m][topic] += 1
                # total number of words assigned to topic j
                self.nwsum[topic] += 1
            # total number of words in document i
            self.ndsum[m] = N

        self.pnewdata = DataSet()
        if self.withrawstrs:
            if self.pnewdata.read_newdata_withrawstrs(self.dir + self.dfile, self.dir + self.wordmapfile):
                print("Fail to read self.new data!\n")
                return 1
        else:
            if self.pnewdata.read_newdata(self.dir + self.dfile, self.dir + self.wordmapfile):
                print("Fail to read self.new data!\n")
                return 1
        self.newM = self.pnewdata.M
        self.newV = self.pnewdata.V

        self.newnw = []  # int*[self.newV]
        for w in range(self.newV):
            newnw_row = []
            for k in range(self.K):
                newnw_row.append(0)
            self.newnw.append(newnw_row)

        self.newnd = []  # int*[self.newM]
        for w in range(self.newM):
            newnd_row = []  # int[K]
            for k in range(self.K):
                newnd_row.append(0)
            self.newnd.append(newnd_row)

        self.newnwsum = []  # int[K]
        for k in range(self.K):
            self.newnwsum.append(0)

        self.newndsum = []  # int[self.newM]
        for k in range(self.newM):
            self.newndsum.append(0)

        self.newz = []  # int*[self.newM]
        for m in range(self.newM):
            self.newz.append(0)

        for m in range(self.pnewdata.M):
            N = self.pnewdata.docs[m].length
            newz_row = []  # int[N]

            # assign values for nw,nd,nwsum, and ndsum
            for n in range(N):
                w = self.pnewdata.docs[m].words[n]
                topic = random.randint(0, self.K-1)
                newz_row.append(topic)

                # number of instances of word i assigned to topic j
                self.newnw[w][topic] += 1
                # number of words in document i assigned to topic j
                self.newnd[m][topic] += 1
                # total number of words assigned to topic j
                self.newnwsum[topic] += 1
            # total number words in document i
            self.newndsum[m] = N
            self.newz[m] = newz_row

        self.newtheta = []  # double*[m]
        for m in range(self.newM):
            newtheta_row = []  # double[K]
            for k in range(self.K):
                newtheta_row.append(0)
            self.newtheta.append(newtheta_row)

        self.newphi = []  # double*[K]
        for k in range(self.K):
            newphi_row = []  # double[self.newV]
            for v in range(self.newV):
                newphi_row.append(0)
            self.newphi.append(newphi_row)

        return 0

    # @tested
    def load_model(self, model_name):
        filename = self.dir + model_name + self.tassign_suffix
        fin = open(filename)
        if not fin:
            print("Cannot open file ", filename, " to load model")
            return 1

        self.z = []
        for n in range(self.M):
            self.z.append([])

        ptrndata = DataSet(self.M)
        ptrndata.V = self.V

        for i in range(self.M):
            line = fin.readline()
            if not line:
                print("Invalid word-topic assignment file, check the number of docs!\n")
                return 1
            strtok = Strtokenizer(line, ' \t\r\n')
            length = strtok.count_tokens()

            words = []
            topics = []
            for j in range(length):
                token = strtok.token(j)
                tok = Strtokenizer(token,':')
                if tok.count_tokens() != 2:
                    print("Invalid word-topic assignment line!\n")
                    return 1
                words.append(int(tok.token(0)))
                topics.append(int(tok.token(1)))

            pdoc = Document(words)
            ptrndata.add_doc(pdoc, i)

            # assign values for z
            for to in range(len(topics)):
                self.z[i].append(0)
            for j in range(len(topics)):
                self.z[i][j] = topics[j]

        self.ptrndata = ptrndata
        fin.close()
        return 0

    # @tested
    def save_model(self, model_name):
        if self.save_model_tassign(self.dir + model_name + self.tassign_suffix):
            return 1
        if (self.save_model_others(self.dir + model_name + self.others_suffix)):
            return 1
        if (self.save_model_theta(self.dir + model_name + self.theta_suffix)):
            return 1
        if (self.save_model_phi(self.dir + model_name + self.phi_suffix)):
            return 1
        if self.twords > 0:
            if (self.save_model_twords(self.dir + model_name + self.twords_suffix)):
                return 1
        return 0

    # @tested
    def save_model_tassign(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for i in range(self.ptrndata.M):
            for j in range(self.ptrndata.docs[i].length):
                tmp = str(self.ptrndata.docs[i].words[j]) + ":" + str(self.z[i][j]) + " "
                fout.write(tmp)
            fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_model_theta(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for i in range(self.M):
            for j in range(self.K):
                fout.write(str(self.theta[i][j]) + " ")
            fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_model_phi(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for i in range(self.K):
            for j in range(self.V):
                fout.write(str(self.phi[i][j]) + " ")
            fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_model_twords(self,filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        if(self.twords > self.V) :
            self.twords = self.V

        for k in range(self.K) :
            words_probs = []
            for w in range(self.V) :
                word_prob = {w:self.phi[k][w]}
                words_probs.append(word_prob)

            # quick sort to word-topic probability
            u = Utils()
            u.quicksort(words_probs,0,len(words_probs)-1)

            tmp = "Topic " + str(k) + "th:\n"
            fout.write(tmp)
            for i in range(self.twords) :
                found = False
                for key in self.id2word.keys() :
                    if list(words_probs[i].keys())[0] == int(key) :
                        found = True
                        break
                if found :
                    tmp = "\t" + str(list(words_probs[i].keys())[0]) + " " + str(list(words_probs[i].values())[0]) + '\n'
                    fout.write(tmp)

        fout.close()
        return 0

    # @tested
    def save_model_others(self,filename):
        fout = open(filename,'w')
        if not fout :
            print("Cannot open file ",filename," to save!\n")
            return 1
        tmp = "alpha=" + str(self.alpha) + '\n'
        fout.write(tmp)
        tmp = "beta=" + str(self.beta) + '\n'
        fout.write(tmp)
        tmp = "ntopics=" + str(self.K) + '\n'
        fout.write(tmp)
        tmp = "ndocs=" + str(self.M) + '\n'
        fout.write(tmp)
        tmp = "nwords=" + str(self.V) + '\n'
        fout.write(tmp)
        tmp = "liter=" + str(self.liter) + '\n'
        fout.write(tmp)
        fout.close()
        return 0

    # @tested
    def save_inf_model(self, model_name):
        if self.save_inf_model_tassign(self.dir + model_name + self.tassign_suffix):
            return 1
        if (self.save_inf_model_others(self.dir + model_name + self.others_suffix)):
            return 1
        if (self.save_inf_model_newtheta(self.dir + model_name + self.theta_suffix)):
            return 1
        if (self.save_inf_model_newphi(self.dir + model_name + self.phi_suffix)):
            return 1
        if self.twords > 0:
            if (self.save_inf_model_twords(self.dir + model_name + self.twords_suffix)):
                return 1
        return 0

    # @tested
    def save_inf_model_tassign(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for i in range(self.pnewdata.M):
            for j in range(self.pnewdata.docs[i].length):
                tmp = str(self.pnewdata.docs[i].words[j]) + ":" + str(self.newz[i][j]) + " "
                fout.write(tmp)
            fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_inf_model_newtheta(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for i in range(self.newM):
            for j in range(self.K):
                fout.write(str(self.newtheta[i][j]) + " ")
            fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_inf_model_newphi(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for i in range(self.K):
            for j in range(self.newV):
                fout.write(str(self.newphi[i][j]) + " ")
            fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_inf_model_twords(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        if (self.twords > self.newV):
            self.twords = self.newV

        for k in range(self.K):
            words_probs = []
            for w in range(self.newV):
                word_prob = {w: self.newphi[k][w]}
                words_probs.append(word_prob)

            # quick sort to word-topic probability
            u = Utils()
            u.quicksort(words_probs, 0, len(words_probs) - 1)

            tmp = "Topic " + str(k) + "th:\n"
            fout.write(tmp)
            for i in range(self.twords):
                found = False
                for key in self.pnewdata._id2id.keys() :
                    found2 = False
                    if list(words_probs[i].keys())[0] == key :
                        found2 = True
                        break
                if not found2 :
                    continue
                else :
                    for i in self.id2word :
                        if self._id2id(list(words_probs[i].keys())[0]) == key:
                            found = True
                            break
                    if found :
                        tmp = "\t" + list(words_probs[i].keys())[0] + " " + list(words_probs[i].values())[0] + '\n'
                        fout.write(tmp)

        fout.close()
        return 0

    # @tested
    def save_inf_model_others(self,filename):
        fout = open(filename,'w')
        if not fout :
            print("Cannot open file ",filename," to save!\n")
            return 1
        tmp = "alpha=" + str(self.alpha) + '\n'
        fout.write(tmp)
        tmp = "beta=" + str(self.beta) + '\n'
        fout.write(tmp)
        tmp = "ntopics=" + str(self.K) + '\n'
        fout.write(tmp)
        tmp = "ndocs=" + str(self.newM) + '\n'
        fout.write(tmp)
        tmp = "nwords=" + str(self.newV) + '\n'
        fout.write(tmp)
        tmp = "liter=" + str(self.inf_liter) + '\n'
        fout.write(tmp)
        fout.close()
        return 0

    # @tested
    def estimate(self):
        if self.twords > 0 :
            da = DataSet()
            da.read_wordmap2(self.dir + self.wordmapfile,self.id2word)

        print("Sampling ",self.niters," iterations!\n")

        last_iter = self.liter
        for self.liter in range(last_iter+1,self.niters+last_iter) :
            print("Iteration ",self.liter," ...\n")

            # for all z_i
            for m in range(self.M) :
                for n in range(self.ptrndata.docs[m].length) :
                    # (z_i) = z[m][n]
                    # sample from p(z_i|z_-i,w)
                    topic = self.sampling(m,n)
                    self.z[m][n] = topic

            if self.savestep > 0 :
                if self.liter % self.savestep == 0 :
                    # saving the model
                    print("Saving the model at iteration ",self.liter," ...\n")
                    self.compute_theta()
                    self.compute_phi()
                    u = Utils()
                    self.save_model(u.generate_model_name(self.liter))

        print("Gibbs sampling completed!\n")
        print("Saving the final model!\n")
        self.compute_theta()
        self.compute_phi()
        self.liter -= 1
        u = Utils()
        self.save_model(u.generate_model_name(-1))

    # @tested
    def sampling(self,m,n):
        # remove z_i from the count variables
        topic = self.z[m][n]
        w = self.ptrndata.docs[m].words[n]
        self.nw[w][topic] -= 1
        self.nd[m][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[m] -= 1

        Vbeta = self.V * self.beta
        Kalpha = self.K * self.alpha

        # do multinomial sampling via cumulative method
        for k in range(self.K) :
            self.p[k] = (self.nw[w][k] + self.beta) / (self.nwsum[k] + Vbeta) * (self.nd[m][k] + self.alpha) / (self.ndsum[m] + Kalpha)

        # cumulate multinomial parameters
        for k in range(self.K) :
            self.p[k] += self.p[k-1]

        # scaled sample because of unnormalized p[]         # 需要核实一下
        u = np.random.rand() * self.p[self.K-1]

        topic = 0
        for i in range(self.K) :
            if self.p[i] > u :
                topic = i
                break

        # add newly estimated z_i to count variables
        self.nw[w][topic] += 1
        self.nd[m][topic] += 1
        self.nwsum[topic] += 1
        self.ndsum[m] += 1

        return topic

    # @tested
    def compute_theta(self):
        for m in range(self.M) :
            for k in range(self.K) :
                self.theta[m][k] = (self.nd[m][k] + self.alpha) / (self.ndsum[m] + self.K * self.alpha)

    # @tested
    def compute_phi(self):
        for k in range(self.K):
            for w in range(self.V):
                self.phi[k][w] = (self.nw[w][k] + self.beta) / (self.nwsum[k] + self.V * self.beta)

    # @not tested
    def inference(self):
        if self.twords > 0 :
            DataSet.read_wordmap2(self.dir + self.wordmapfile,self.id2word)

        print("Sampling ",self.niters,' iterations for inference!\n')

        for self.inf_liter in range(1,self.niters+1) :
            print("Iteration ",self.inf_liter," ...\n")

            # for all newz_i
            for m in range(self.newM) :
                for n in range(self.pnewdata.docs[m].length) :
                    # newz_i = newz[m][n]
                    # sample from p(z_i|z_-i,w)
                    topic = self.inf_sampling(m,n)
                    self.newz[m][n] = topic

        print("Gibbs sampling for inference completed!\n")
        print("Saving the inference outputs!\n")
        self.compute_newtheta()
        self.compute_newphi()
        self.inf_liter -= 1
        self.save_inf_model(self.dfile)

    # @not tested
    def inf_sampling(self,m,n):
        # remove z_i from the count variables
        topic = self.newz[m][n]
        w = self.pnewdata.docs[m].words[n]
        _w = self.pnewdata._docs[m].words[n]
        self.newnw[_w][topic] -= 1
        self.newnd[m][topic] -= 1
        self.newnwsum[topic] -= 1
        self.newndsum[m] -= 1

        Vbeta = self.V * self.beta
        Kalpha = self.K * self.alpha

        # do multinomial sampling via cumulative method
        for k in range(self.K) :
            self.p[k] = (self.nw[w][k] + self.newnw[_w][k] + self.beta) / (self.nwsum[k] + self.newnwsum[k] + Vbeta) * (self.newnd[m][k] + self.alpha) / (self.newndsum[m] + Kalpha)

        # cumulate multinomial parameters
        for k in range(self.K) :
            self.p[k] += self.p[k-1]

        # scaled sample because of unnormalized p[]
        u = np.random.rand() * self.p[self.K-1]

        for topic in range(self.K) :
            if self.p[topic] > u :
                break

        # add newly estimated z_i to count variables
        self.nw[_w][topic] += 1
        self.nd[m][topic] += 1
        self.nwsum[topic] += 1
        self.ndsum[m] += 1

        return topic

    # @not tested
    def compute_newtheta(self):
        for m in range(self.newM) :
            for k in range(self.K) :
                self.newtheta[m][k] = (self.newnd[m][k] + self.alpha) / (self.newndsum[m] + self.K * self.alpha)

    # @not tested
    def compute_newphi(self):
        for k in range(self.K):
            for w in range(self.newV):
                found = False
                for key in self.pnewdata._id2id.keys() :
                    if key == w :
                        found = True
                if found:
                    self.newphi[k][w] = (self.nw[self.pnewdata._id2id[w]][k] + self.newnw[w][k] + self.beta) / (self.nwsum[k] + self.newnwsum[k] + self.V * self.beta)