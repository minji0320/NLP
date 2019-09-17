from collections import Counter
wordDict = Counter()

count = {}



f = open('Doc01.txt', 'r')
words = [word.strip('.,!?/()&-') for word in f.read().lower().split()]
f = open('Doc02.txt', 'r')
words += [word.strip('.,!?/()&-') for word in f.read().lower().split()]
f = open('Doc03.txt', 'r')
words += [word.strip('.,!?/()&-') for word in f.read().lower().split()]
f = open('Doc04.txt', 'r')
words += [word.strip('.,!?/()&-') for word in f.read().lower().split()]
f = open('Doc05.txt', 'r')
words += [word.strip('.,!?/()&-') for word in f.read().lower().split()]
f = open('Doc06.txt', 'r')
words += [word.strip('.,!?/()&-') for word in f.read().lower().split()]
f = open('Doc07.txt', 'r')
words += [word.strip('.,!?/()&-') for word in f.read().lower().split()]
f = open('Doc08.txt', 'r')
words += [word.strip('.,!?/()&-') for word in f.read().lower().split()]
f = open('Doc09.txt', 'r')
words += [word.strip('.,!?/()&-') for word in f.read().lower().split()]
f = open('Doc10.txt', 'r')
words += [word.strip('.,!?/()&-') for word in f.read().lower().split()]

for word, index in Counter(words).most_common(30):
    count[word] = index

print(count)

import matplotlib.pyplot as plt
plt.plot(count.keys(), count.values())
plt.xlabel('Words')
plt.ylabel('Counting')
plt.title('Top 30 words')
plt.show()
