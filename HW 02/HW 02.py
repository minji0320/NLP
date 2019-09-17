from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import math
from operator import itemgetter
import random

#
# 데이터 정리 및 단어들의 Inverted Index 구하기
#

wordDict = Counter()
inverted = {}

stopwords = "a,able,about,across,after,all,almost,also,am,among,an,and,any,\
             are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,\
             does,either,else,ever,every,for,from,get,got,had,has,have,he,\
             her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,\
             let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,\
             off,often,on,only,or,other,our,own,rather,said,say,says,she,should,\
             since,so,some,than,that,the,their,them,then,there,these,they,this,\
             tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,\
             who,whom,why,will,with,would,yet,you,your"
stopword = stopwords.split(',')

def index(f,i):
    words = [word.strip('.,!?/()[]#-":') for word in f.read().lower().split()]
    clean_words = []
    for w in words:
        if w not in stopword:           # stopword 제거
                if not w.isdigit():     # 숫자 제거
                    if w not in '':     # '' 제거
                        clean_words.append(w) 
    
    for word, index in Counter(clean_words).most_common():
        freq = inverted.setdefault(word,[])
        freq.append([i,index])

f = open('Doc01.txt', 'r')
index(f,1)
f = open('Doc02.txt', 'r')
index(f,2)
f = open('Doc03.txt', 'r')
index(f,3)
f = open('Doc04.txt', 'r')
index(f,4)
f = open('Doc05.txt', 'r')
index(f,5)
f = open('Doc06.txt', 'r')
index(f,6)
f = open('Doc07.txt', 'r')
index(f,7)
f = open('Doc08.txt', 'r')
index(f,8)
f = open('Doc09.txt', 'r')
index(f,9)
f = open('Doc10.txt', 'r')
index(f,10)

terms = []
for i in inverted.keys():
    terms.append(i)
num_terms = len(terms)

freq = []
for i in inverted.values():
    freq.append(i)


#
# TF-IDF 행렬 구하기
#

tf = np.zeros((10,num_terms), dtype = int)
for i in range(num_terms):
    for j in range(len(freq[i])):
        tf[freq[i][j][0]-1,i] = freq[i][j][1]

df = []
for i in range(num_terms):
    count = 0
    for j in range(10):
        if tf[j,i] != 0:
            count += 1
    df.append(count)

tf_idf = np.zeros((10,num_terms), dtype = float)
for i in range(num_terms):
    tf_idf[:,i] = tf[:,i] * math.log(10/df[i],2)


#
# Cosine Similarity 계산하기
#

def quety_index_weight(query_words):
    index = []
    for w in query_words: 
        for i in range(num_terms):
            if w == terms[i]:
                index.append(i)
    index_count = Counter(index)
    query_index = []
    for i in index_count.keys():
        query_index.append(i)
    query_weight = []
    for i in index_count.values():
        query_weight.append(i)
    return query_index, query_weight

def cos_sim(index, doc_num, query_weight, weight):
    doc_weight = []
    for i in index:
        doc_weight.append(weight[doc_num-1,i])
    if sum(np.array(query_weight)*np.array(doc_weight)) == 0 : sim = 0
    else: sim = sum(np.array(query_weight)*np.array(doc_weight))\
                    / (np.linalg.norm(query_weight, axis=0, ord=2)\
                    * np.linalg.norm(doc_weight, axis=0, ord=2))
    return sim


#
# Sim, Rank 계산 함수
#

def sim_function(index, query_weight,tf_idf):
    sim = np.empty((10,2),dtype = float)
    for i in range(10):
        sim[i,0] = i+1
        sim[i,1] = cos_sim(index, i, query_weight, tf_idf)
    idx = np.argsort(sim[:,1], axis = 0)[::-1]
    sim = sim[idx,:]
    rank = []
    for i in range(10):
        if sim[i,1] != 0:
            rank.append(int(sim[i,0])) 
    return sim, rank


#
# 관련성 계산 함수
#
def rel_function(tf_idf):
    rel = []
    count = np.zeros((10),dtype = int)
    for i in range(10):
        for j in index:
            if tf_idf[i,j] > 5: count[i] += 1
        if count[i] != 0:
            rel.append(i+1)
    return rel



#
# AvgPrec 계산 함수
#
def AvgPrec_function(rank,rel):
    count = 0
    sum = 0
    for i in range(len(rank)):
        for j in range(len(rel)):
            if rank[i] == rel[j]:
                count +=1
                sum += count/(i+1)
    if sum == 0 : AvgPrec = 0
    else : AvgPrec = (1/count)*sum
    return AvgPrec



#
# 함수 이용하여 Text Ranking 구현하기
#
sum_AvgPrec = 0

for i in range(5):
    k = 30
    query_words = ["beast"]
    index, query_weight = quety_index_weight(query_words)

    sim, rank = sim_function(index, query_weight,tf_idf)
    print("Rank :", rank)


    rel = rel_function(tf_idf)
    print("Rel_Doc :",rel)

    AvgPrec = np.empty((5),dtype = float)
    
    AvgPrec[i] = AvgPrec_function(rank,rel)
    sum_AvgPrec += AvgPrec[i]
    print("Avg. Prec :", AvgPrec[i],"\n")

MAP = sum_AvgPrec/5
print("MAP :",MAP)
    

