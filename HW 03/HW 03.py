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
    words = [word.strip('.,!?/()[]#-":;') for word in f.read().lower().split()]
    clean_words = []
    for w in words:
        if w not in stopword:           # stopword 제거
                if not w.isdigit():     # 숫자 제거
                    if w not in '':     # '' 제거
                        clean_words.append(w) 
    # SDV 구할때 행렬 크기가 커서 에러 발생하여 문서들의 top 1000의 단어들만 추출
    for word, index in Counter(clean_words).most_common(1000):
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

tf_idf = tf_idf.T   # 이전의 과제에서 10*term_size로 생성하여 변환해줌


#
# SVD를 이용한 LSA (k = 10)
#
U, s, V = np.linalg.svd(tf_idf, full_matrices = True)

S = np.zeros((len(s),len(s))) # s는 eigenvalue 리스트여서 대각행렬 S로 변환
for i in range(len(s)):
    S[i][i] = s[i]
U = U[:,:10]    # U의 크기를 4207*10으로 축소

A = U@S@V
A = A.T     # 이전의 과제 폼을 이용하기 위해 다시 전치시켜줌


#
# Cosine Similarity 계산하기
#

def quety_weight_function(query_words):
    index = []
    for w in query_words: 
        for i in range(num_terms):
            if w == terms[i]:
                index.append(i)
    index_count = Counter(index)

    query_weight = np.zeros((num_terms), dtype = int)
    for i in index_count.keys():
        for j in index_count.values():
            query_weight[i] = j

    query_index = []
    for i in index_count.keys():
        query_index.append(i)
        
    return query_index, query_weight

def cos_sim(D, Q):
    sim = np.empty((10,2),dtype = float)
    for i in range(10):
        sim[i,0] = i+1
        sim[i,1] = sum(D[i,:]*Q)/(np.linalg.norm(D[i,:],ord=2)*np.linalg.norm(Q,ord=2))
    idx = np.argsort(sim[:,1], axis = 0)[::-1]
    sim = sim[idx,:]
    return sim


#
# Rank 계산 함수
#

def rank_function(sim):
    rank = []
    for i in range(10):
        if sim[i,1] > 0:
            rank.append(int(sim[i,0])) 
    return rank


#
# 관련성 계산 함수
#
def rel_function(A):
    rel = []
    count = np.zeros((10),dtype = int)
    for i in range(10):
        for j in index:
            if A[i,j] >= 20: count[i] += 1
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

# 문서 2, 3에서 많이 나올 단어 선정
# 결과 : 각 영화의 주인공 이름을 쿼리로 택하여서 그 영화들의 sim이 매우 높게 나옴 - 정확
query_words = ["miguel", "deadpool"]   
query_weight = np.zeros((num_terms), dtype = int)
index, query_weight = quety_weight_function(query_words)
print("Query :",query_words)

sim = cos_sim(A,query_weight)
rank = rank_function(sim)
print("Sim :")
for i in range(10):
    print(" ",int(sim[i,0]),"\t",sim[i,1])
print("Rank : ", rank)

rel = rel_function(A)
print("Rel_Doc :",rel)

AvgPrec = np.empty((5),dtype = float)
for i in range(5):
    AvgPrec[i] = AvgPrec_function(rank,rel)
print("Avg. Prec :", AvgPrec[i],"\n\n")


# 문서 7에서 많이 나올 단어 선정
# 결과 : 7이 1순위로 랭크됨 - 정확
query_words = ["space", "earth"]
query_weight = np.zeros((num_terms), dtype = int)
index, query_weight = quety_weight_function(query_words)
print("Query :",query_words)

sim = cos_sim(A,query_weight)
rank = rank_function(sim)
print("Sim :")
for i in range(10):
    print(" ",int(sim[i,0]),"\t",sim[i,1])
print("Rank : ", rank)

rel = rel_function(A)
print("Rel_Doc :",rel)

AvgPrec = np.empty((5),dtype = float)
for i in range(5):
    AvgPrec[i] = AvgPrec_function(rank,rel)
print("Avg. Prec :", AvgPrec[i],"\n\n")

# 문서 5에서 많이 나올 단어 선정
# 결과 : 문서 8에서 piano라는 단어가 많이 나와서 5가 1순위, 8이 2순위로 랭크됨
query_words = ["piano","jazz"]  
query_weight = np.zeros((num_terms), dtype = int)
index, query_weight = quety_weight_function(query_words)
print("Query :",query_words)

sim = cos_sim(A,query_weight)
rank = rank_function(sim)
print("Sim :")
for i in range(10):
    print(" ",int(sim[i,0]),"\t",sim[i,1])
print("Rank : ", rank)

rel = rel_function(A)
print("Rel_Doc :",rel)

AvgPrec = np.empty((5),dtype = float)
for i in range(5):
    AvgPrec[i] = AvgPrec_function(rank,rel)
print("Avg. Prec :", AvgPrec[i],"\n\n")

