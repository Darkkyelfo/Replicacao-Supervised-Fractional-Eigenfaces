'''
Created on Aug 14, 2017

@author: raul
'''
from copy import deepcopy
from random import shuffle
from sklearn import preprocessing

class Base(object):
    
    def __init__(self, classes=[],atributos=[],nome=""):
        self.nome = nome
        self.tiposClasses = []
        self.classesOri = deepcopy(classes)
        self.classes = self.classesOri
        self.atributos = deepcopy(atributos)
        self.qtElementos = len(self.classes)
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self._findTiposClasses()
    
    def _findTiposClasses(self):
        self.tiposClasses = []
        for e in self.classes:
            if(e not in self.tiposClasses):
                self.tiposClasses.append(e)

    def getSubBaseClasse(self,indice):
        subClasse = []
        subAtributos = []
        for i,e in enumerate(self.classesOri):
            if(e == self.tiposClasses[indice]):
                subClasse.append(e)
                subAtributos.append(self.atributos[i])
        return Base(subClasse,subAtributos)
    
    def toClasseNumericas(self,mapeamento = {}):
        self.classes = []
        if len(mapeamento)==0:
            for i in self.classesOri:
                self.classes.append(self.tiposClasses.index(i))
        else:
            for i in self.classesOri:
                self.classes.append(mapeamento[i])
    
    #embaralha os elementos da base
    def embaralharBase(self):
        c = list(zip(self.classes, self.atributos,self.classesOri))
        shuffle(c)
        self.classes,self.atributos,self.classesOri = zip(*c)
    
    def gerarBaseComMenosAtr(self,qtAtributos):
        novosAtr = []
        for i in self.atributos:
            novosAtr.append(i[0:qtAtributos])
        return Base(self.classesOri,novosAtr)
    
    def normalizar(self):
        self.atributosOri = deepcopy(self.atributos)
        self.atributos = self.min_max_scaler.fit_transform(self.atributos)
    
    def desnormalizar(self):
        self.atributos = self.atributosOri
        
        
    def copy(self):
        return Base(self.classes,self.atributos)
    
    def gerarSubBases(self,qtSubBases):
        self.embaralharBase()
        bases = []
        div = int(self.qtElementos/qtSubBases)
        ant = 0
        for i in range(qtSubBases):
            bases.append(Base(self.classes[ant:div],self.atributos[ant:div],self.posicoes[ant:div]))
            ant = div
            div = div + div
        return bases

        
class BaseImg(Base):   
    
    def __init__(self,classes=[],atributos=[],matrizImgs=[],nome=""):
        super().__init__(classes, atributos, nome)
        self.matrizImgs = deepcopy(matrizImgs)
        self.qtPixels = self.matrizImgs[0].shape[0]*self.matrizImgs[0].shape[1]
        self.linhas = self.matrizImgs[0].shape[0]
        self.colunas = self.matrizImgs[0].shape[1]
            
        
        

                
        
        