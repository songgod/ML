import numpy as np
import math

def loaddataset():
    postinglist = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classvec = [0, 1, 0, 1, 0, 1]
    return postinglist, classvec


def createvocablist(dataset):
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set(document)
    return list(vocabset)


def setofwords2vec(vocablist, inputset):
    returnvec = [0]*len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!", word)
    return returnvec


def trainnb0(trainmatrix, traincategory):
    numoftraindocs = len(trainmatrix)
    numofwords = len(trainmatrix[0])
    pabusive = sum(traincategory)/float(numoftraindocs)
    p0num = np.ones(numofwords)
    p1num = np.ones(numofwords)
    p0denom = 2.0
    p1denom = 2.0
    for i in range(numoftraindocs):
        if traincategory[i] == 1:
            p1num += trainmatrix[i]
            p1denom += sum(trainmatrix[i])
        else:
            p0num += trainmatrix[i]
            p0denom += sum(trainmatrix[i])
    p1vec = np.log(p1num/p1denom)
    p0vec = np.log(p0num/p0denom)
    return p0vec, p1vec, pabusive


def classifynb(vec2classify, p0vec, p1vec, pclass1):
    p1 = sum(vec2classify * p1vec) + np.log(pclass1)
    p0 = sum(vec2classify * p0vec) + np.log(1-pclass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testclassifynb():
    listposts, listclasses = loaddataset()
    listvocabset = createvocablist(listposts)
    trainmat = []
    for post in listposts:
        trainmat.append(setofwords2vec(listvocabset, post))
    p0v, p1v, pab = trainnb0(trainmat, listclasses)
    testentry = ['love', 'my', 'dalmation']
    thisdoc = np.array(setofwords2vec(listvocabset, testentry))
    print(testentry, 'classified as :', classifynb(thisdoc, p0v, p1v, pab))
    testentry = ['stupid', 'garbage']
    thisdoc = np.array(setofwords2vec(listvocabset, testentry))
    print(testentry, 'classfied as :', classifynb(thisdoc, p0v, p1v, pab))


if __name__ == '__main__':
    testclassifynb()
