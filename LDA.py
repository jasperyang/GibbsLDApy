#
# Copyright (C) 2007 by
#
# 	Xuan-Hieu Phan
#	hieuxuan@ecei.tohoku.ac.jp or pxhieu@gmail.com
# 	Graduate School of Information Sciences
# 	Tohoku University
#
# GibbsLDA++ is a free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# GibbsLDA++ is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GibbsLDA++; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#

#
# References:
# + The Java code of Gregor Heinrich (gregor@arbylon.net)
#   http://www.arbylon.net/projects/LdaGibbsSampler.java
# + "Parameter estimation for text analysis" by Gregor Heinrich
#   http://www.arbylon.net/publications/text-est.pdf
#

#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: LDA.py
@time: 3/9/17 5:34 PM
@desc: gate to run LDA!
'''

from Model import *
import sys
import Constants

def show_help() :
    print("Command line usage:\n")
    print("\tlda -est -alpha <double> -beta <double> -ntopics <int> -niters <int> -savestep <int> -twords <int> -dfile <string>\n")
    print("\tlda -estc -dir <string> -model <string> -niters <int> -savestep <int> -twords <int>\n")
    print("\tlda -inf -dir <string> -model <string> -niters <int> -twords <int> -dfile <string>\n")
    # print("\tlda -inf -dir <string> -model <string> -niters <int> -twords <int> -dfile <string> -withrawdata\n")

if __name__ == '__main__' :
    lda = Model()

    if (lda.init(len(sys.argv)-1, sys.argv)) :
        show_help();

    if (lda.model_status == Constants.MODEL_STATUS_EST or lda.model_status == Constants.MODEL_STATUS_ESTC) :
        # parameter estimation
        lda.estimate()

    if (lda.model_status == Constants.MODEL_STATUS_INF) :
        # do inference
        lda.inference()

