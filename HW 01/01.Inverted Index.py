from collections import Counter
wordDict = Counter()
inverted = {}

def index(f,i):
    words = [word.strip('.,!?/()#-') for word in f.read().lower().split()]

    for word, index in Counter(words).most_common(100):
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

print(inverted)


