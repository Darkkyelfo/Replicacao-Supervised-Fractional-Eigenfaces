'''
Created on Aug 29, 2017

@author: raul
'''
import numpy as np
from base import Base
import math
import copy
from numba import jit

class PCA(object):
    
    def run (self,base1,k="T"):
        if(k=="T"):
            k = len(base1.atributos[0])

        copia = np.array(copy.deepcopy(base1.atributos))
        autoVectors = self.autoVectors[0:k]
        #novosAtributos = np.dot(copia,np.array(autoVectors).T)
        novosAtributos = self._projetar(copia, autoVectors)
        return (Base(base1.classes,novosAtributos))
    
    def fit(self,bTreino):
        copia = np.array(copy.deepcopy(bTreino.atributos))
        cov = np.cov(copia.T)
        autoValues,autoVectors = np.linalg.eig(cov)
        autoVectors = autoVectors.T
        self.autoValues,self.autoVectors = zip(*sorted(zip(autoValues, autoVectors),reverse=True))
    
    
    def _projetar(self,atributorsOri,autoVetores):
        return np.dot(atributorsOri,np.array(autoVetores).T)
    
class PCA_SCORE(PCA):
    
    def fit(self,base1):
        super().fit(base1)
        base = super().run(base1, len(base1.classes))
        subBases = []
        medias = [0]*len(base.tiposClasses)
        for i,e in enumerate(medias):
            subBases.append(base.getSubBaseClasse(i))
            medias[i] = np.mean(subBases[i].atributos,axis=0)
        scores = self.__score(medias)
        self.scores,self.autoVectors = zip(*sorted(zip(scores, self.autoVectors ),reverse=True))
    
    @jit
    def __score(self,medias):
        scores = []
        for i in range(len(medias[0])):
            s = 0
            if(self.autoValues[i] != 0):
                s = math.fabs((self.__subTrairVetor(medias, i)))/self.autoValues[i]
            scores.append(s)
        return scores
    #(m1-mu)^2/lamb + (m2-mu)^2/lamb + (m3-mu)^2/lamb
    #(m1-m2)/lamb + (m1-m3)/lamb
    #testar esse tipo de média
    def __subTrairVetor(self,lista,indice):
        resultado = 0
        for i in lista:
            resultado = i[indice] - resultado
        
        return resultado  

class FractionalPCA(PCA):
    
    def fit(self,bTreino,r=0.01):
        copia = np.array(copy.deepcopy(bTreino.atributos))
        media = np.mean(copia,axis=0)
        sub = np.array(np.power(copia,r)-np.power(media,r))
        fracCov = np.dot(sub.T,sub)/len(copia[0])
        autoValues,autoVectors = np.linalg.eig(fracCov)
        autoVectors = autoVectors.T
        self.autoValues,self.autoVectors = zip(*sorted(zip(autoValues, autoVectors),reverse=True))

class PCAWhitening(PCA):
    
    def run (self,base1,k="T"):
        if(k=="T"):
            k = len(base1.atributos[0])

        copia = np.array(copy.deepcopy(base1.atributos))
        autoVectors = self.autoVectors[0:k]
        novosAtributos = np.dot(copia,np.array(autoVectors).T)
        for i,a in enumerate(novosAtributos):
            for j,atr in enumerate(a): 
                novosAtributos[i][j] = atr/math.sqrt(self.autoValues[j])
        
        return (Base(base1.classes,novosAtributos))
    

class DualSupervisedPCA(PCA):
    
    def _gerarMatrizCov(self,base ,copia, n):
        self.delta = self._gerarMatrizDelta(base, copia, n)
        mIdentidade = np.identity(base.qtElementos)
        h = np.ones(base.qtElementos,1)
        H = mIdentidade - 1/(base.qtElementos)*(h.dot(h.T))
        S = self.delta.dot(H).dot(copia.dot(copia.T)).dot(H).dot(self.delta.T)
        return S
        
    def _gerarMatrizDelta(self,base,copia,n):
        qtClasses = len(base.tiposClasses)
        delta = np.zeros((qtClasses, base.qtElementos))
        for classe in range(qtClasses):
            for coluna in range(base.qtElementos):
                if(classe == base.classe[coluna]):
                    delta[classe][coluna] = 1
        return delta
    
    