'''
Created on 14 de nov de 2017

@author: raul
'''
from pca import PCA
from copy import deepcopy
import numpy as np
from numba import jit
import math

class Eigenfaces(PCA):
    '''
        Classe que representa a técnica Eigenfaces
    '''
    
    #@jit
    def fit(self,bTreino):
        copia = np.array(deepcopy(bTreino.atributos),dtype='f')
        self.media = np.mean(copia,axis=0)
        n = len(copia)
        copia = copia.T
        self._gerarMatrizSub(copia)
        cov = self._gerarMatrizCov(copia, n)#matriz de covariancia
        autoValues,autoVectors = np.linalg.eig(cov)
        autoVectors = np.array(autoVectors).T
        autoValues,autoVectors =  zip(*sorted(zip(autoValues, autoVectors),reverse=True))
        self._encontrarAutovetores(copia.T,autoVectors,autoValues, n)

    @jit
    def _gerarMatrizSub(self,matriz):
        colunas = len(matriz[0])
        linhas = len(matriz)
        for j in range(colunas):
            for i in range(linhas):
                #realiza a subtracao do valor - a media ambos elevador a r.
                matriz[i][j] = matriz[i][j] - self.media[i]
                
    def _encontrarAutovetores(self,sub,autoVectores,autoValores,n):
        self.autoVectors = []
        for i,e in enumerate(autoVectores):
            if(autoValores[i]<=0):
                part = 0
            else:
                part = (1/((n*autoValores[i])**(1/2)))
            autoVetor = part*sub.T.dot(e)
            self.autoVectors.append(autoVetor)
        self.autoVectors = np.array(self.autoVectors)
        
    
    def _gerarMatrizCov(self,copia,n):
        return (1/n)*(copia.T.dot(copia))
    
    '''
    def _projetar(self, atributorsOri, autoVetores):
        atrR = []
        media = np.mean(atributorsOri,axis=0)
        
        for i in atributorsOri:
            temp = []
            for ind,atr in enumerate(i):
                #temp.append(math.pow(atr, self.r) - math.pow(media[ind],self.r))
                temp.append(atr - self.media[ind])
            atrR.append(temp)
        return super()._projetar(atrR,autoVetores)
    '''
class FractionalEigenfaces(Eigenfaces):
    
    def fit(self,bTreino,r=0.01):
        self.r = r
        super().fit(bTreino)

    @jit
    def _gerarMatrizSub(self,matriz):
        colunas = len(matriz[0])
        linhas = len(matriz)
        for j in range(colunas):
            for i in range(linhas):
                #realiza a subtracao do valor - a media ambos elevador a r.
                matriz[i][j] = math.pow(matriz[i][j],self.r) - math.pow(self.media[i],self.r)
    
    def _projetar(self, atributorsOri, autoVetores):
        atrR = []
        #media = np.mean(atributorsOri,axis=0)
        for i in atributorsOri:
            temp = []
            for ind,atr in enumerate(i):
                #temp.append(math.pow(atr, self.r) - math.pow(media[ind],self.r))
                temp.append(math.pow(atr, self.r) - math.pow(self.media[ind],self.r))
            atrR.append(temp)
        return super()._projetar(atrR,autoVetores)

class SupervisedFractionalEigenfaces(FractionalEigenfaces):
    
    def fit(self,bTreino,r=0.1):
        self.base = bTreino
        super().fit(bTreino, r)
        
    def _gerarMatrizCov(self,copia, n):
        self.delta = self._gerarMatrizDelta(self.base)
        S = self.delta.dot(copia.T).dot(copia).dot(self.delta.T)
        return S

    @jit
    def _gerarMatrizDelta(self,base):
        qtClasses = len(base.tiposClasses)
        delta = np.zeros((qtClasses, base.qtElementos))
        for classe in base.tiposClasses:
            for coluna in range(base.qtElementos):
                if(classe == base.classes[coluna]):
                    delta[classe][coluna] = 1
        return delta
    
    def _encontrarAutovetores(self,sub,autoVectores,autoValores,n):
        self.autoVectors = []
        for i,e in enumerate(autoVectores):
            if(autoValores[i]<=0):
                part = (1/((-n*autoValores[i])**(1/2)))
            else:
                part = (1/((n*autoValores[i])**(1/2)))
            autoVetor = part*sub.T.dot(self.delta.T).dot(e)
            self.autoVectors.append(autoVetor)
        self.autoVectors = np.array(self.autoVectors)
