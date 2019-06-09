def loadData():
    wordlist=[['my','name','is','David'],
              ['you','are','stupid'],
              ['my','boyfriend','is','NB'],
              ['you','looks','very','smart','i','like','you','very','much']]
    classList=[0,1,1,0]
    return wordlist,classList

import numpy as np
def createVocabList(wordList):
    vocabSet=set()
    for item in wordList:
        vocabSet=vocabSet | set(item)
    vocabList=list(vocabSet)
    return vocabList

def setToVector(vocabList,inputSet):
    vector=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vector[vocabList.index(word)]=1
        else:
            print('this world:%s is not in my vocabulary'%word)
    return vector

def bagToVector(vocabList,inputSet):
    vector=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vector[vocabList.index(word)]+=1
        else:
            print('this world:%s is not in my vocabulary'%word)
    return vector

def trainNB(trainMat,classList):
    numitem=len(trainMat)
    numwords=len(trainMat[0])
    p_spam=sum(classList)/numitem
    p1Num=np.ones(numwords)
    p0Num=np.ones(numwords)
    p1Denom=numwords
    p0Denom=numwords
    for i in range(numitem):
        if classList[i]==1:
            p1Num+=trainMat[i]
            p1Denom+=sum(trainMat[i])
        else:
            p0Num+=trainMat[i]
            p0Denom+=sum(trainMat[i])
    p0V=np.log(p0Num/p0Denom)
    p1V=np.log(p1Num/p1Denom)
    return p0V,p1V,p_spam

def classifyNB(vec2Claasify,p0Vec,p1Vec,pClass):
    p1=sum((vec2Claasify*p1Vec))+np.log(pClass)
    p0=sum((vec2Claasify*p0Vec))+np.log(1-pClass)
    if p1>p0:
        return 1
    else:
        return 0

def testNB():
    wordList,classList=loadData()
    vocabList=createVocabList(wordList)
    trainMat=[]
    for words in wordList:
        trainMat.append(setToVector(vocabList,words))
    p0V,p1V,p_spam=trainNB(trainMat,classList)
    testEntry = ['I', 'like', 'you']
    thisDoc = np.array(setToVector(vocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, p_spam))
    testEntry = ['stupid']
    thisDoc = np.array(setToVector(vocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, p_spam))

testNB()
