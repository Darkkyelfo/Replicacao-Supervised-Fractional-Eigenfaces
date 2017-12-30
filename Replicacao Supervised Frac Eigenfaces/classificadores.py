'''
Created on 15 de out de 2017

@author: raul
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
def classicarKNN(train_atr,train_classes,test_atr,teste_classe,n=1):

    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(train_atr,train_classes)
    knnPredict = knn.predict(test_atr)
    erroKnn = (1-accuracy_score(teste_classe,knnPredict))
    
    return erroKnn

def naiveBayes(train_atr,train_classes,test_atr,teste_classe):
    clf = GaussianNB()
    clf.fit(train_atr, train_classes)
    
    pred = clf.predict(test_atr)
    erroNaiveCli = (1-accuracy_score(teste_classe,pred))
    
    return erroNaiveCli

def arvoreDecisao(train_atr,train_classes,test_atr,teste_classe,mimNodes = 10):
    tree = DecisionTreeClassifier(min_samples_leaf=mimNodes)
    tree.fit(train_atr,train_classes)
    erroArvoreCli = (1-accuracy_score(teste_classe,tree.predict(test_atr)))
    return erroArvoreCli

def dlFisher(train_atr,train_classes,test_atr,teste_classe):
    clf = LDA()
    clf.fit(train_atr,train_classes)
    pred = clf.predict(test_atr)
    return (1-accuracy_score(teste_classe,pred))
    
    