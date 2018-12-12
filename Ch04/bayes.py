import numpy as np
import random


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
            returnvec[vocablist.index(word)] += 1
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


def textparse(bigstring):
    import re
    listtokens = re.split(r'\W*', bigstring)
    return [tok.lower() for tok in listtokens if len(tok) > 2]


def spamtest():
    docklist = []
    classlist = []
    fulltext = []
    for i in range(1, 26):
        wordlist = textparse(open('email/spam/%d.txt' % i).read())
        docklist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(1)
        wordlist = textparse(open('email/ham/%d.txt' % i).read())
        docklist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(0)
    vacablist = createvocablist(docklist)
    trainingset = list(range(50))
    testset = []
    for i in range(10):
        randindex = int(random.uniform(0, len(trainingset)))
        testset.append(trainingset[randindex])
        del(trainingset[randindex])
    trainingmat = []
    trainingclasses = []
    for docindex in trainingset:
        trainingmat.append(setofwords2vec(vacablist, docklist[docindex]))
        trainingclasses.append(classlist[docindex])
    p0v, p1v, pspam = trainnb0(trainingmat, trainingclasses)
    errocount = 0
    for docindex in testset:
        wordvector = setofwords2vec(vacablist, docklist[docindex])
        if classifynb(wordvector, p0v, p1v, pspam) != classlist[docindex]:
            errocount += 1
    print('the eroor rate is:', float(errocount)/len(testset))


if __name__ == '__main__':
    #testclassifynb()
    spamtest()
