'''
Created on 30 de dez de 2017

@author: raul1
'''
from classificadores import classicarKNN
from imagemparabase import imgsParaBase
from sklearn.model_selection import train_test_split
from eigenfaces import SupervisedFractionalEigenfaces,Eigenfaces,FractionalEigenfaces
from base import Base

def execucaoSupFracEigen(base,arq,qtCla=15,r=0.01,hold=10):
    caminhoArq = "Resultados/%s"%arq
    arqSave = open(caminhoArq,"w")
    arqSave.close()
    arqSave = open(caminhoArq,"a")
    resultados = "atr,acerto\n" 
    arqSave.write(resultados)
    #Bases
    if(base=="georgia"):
        baseAtual = imgsParaBase("Bases/%s"%base,qtClasses=qtCla,dirClasse = "s",tipoArq = "jpg")
    else:
        baseAtual = imgsParaBase("Bases/%s"%base,qtClasses=qtCla)
    #EigenFace
    for j in range(0,41,5):
        erro = 0
        for i in range(hold):
            eigen = SupervisedFractionalEigenfaces()
            train_atr, test_atr, train_classes, test_classes = train_test_split(baseAtual.atributos, baseAtual.classes, test_size=0.5, random_state=i) 
            bTreino = Base(train_classes,train_atr)
            bTeste = Base(test_classes,test_atr)
            eigen.fit(bTreino,r)
            if(j==0):
                j=1
            bTreino = eigen.run(bTreino,j)
            bTeste = eigen.run(bTeste,j)
            erro = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes,1) + erro
        resultados = "erro: %s,%s\n"%(j,(erro/hold))
        print(resultados)
        arqSave.write(resultados)
    arqSave.close()
    

def execucaoEigen(base,arq,qtCla=15,hold=10):
    caminhoArq = "Resultados/%s"%arq
    arqSave = open(caminhoArq,"w")
    arqSave.close()
    arqSave = open(caminhoArq,"a")
    resultados = "atr,acerto\n" 
    arqSave.write(resultados)
    #Bases
    #Bases
    if(base=="georgia"):
        baseAtual = imgsParaBase("Bases/%s"%base,qtClasses=qtCla,dirClasse = "s",tipoArq = "jpg")
    else:
        baseAtual = imgsParaBase("Bases/%s"%base,qtClasses=qtCla)
    #EigenFace
    for j in range(0,41,5):
        erro = 0
        for i in range(hold):
            eigen = Eigenfaces()
            train_atr, test_atr, train_classes, test_classes = train_test_split(baseAtual.atributos, baseAtual.classes, test_size=0.5, random_state=i) 
            bTreino = Base(train_classes,train_atr)
            bTeste = Base(test_classes,test_atr)
            eigen.fit(bTreino)
            if(j==0):
                j=1
            bTreino = eigen.run(bTreino,j)
            bTeste = eigen.run(bTeste,j)
            erro = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes,1) + erro
        resultados = "%s,%s\n"%(j,1 - (erro/hold))
        print(resultados)
        arqSave.write(resultados)
    arqSave.close()
    

def execucaoFracEigen(base,arq,qtCla=15,r=0.01,hold=10):
    caminhoArq = "Resultados/%s"%arq
    arqSave = open(caminhoArq,"w")
    arqSave.close()
    arqSave = open(caminhoArq,"a")
    resultados = "atr,acerto\n" 
    arqSave.write(resultados)
    #Bases
    if(base=="georgia"):
        baseAtual = imgsParaBase("Bases/%s"%base,qtClasses=qtCla,dirClasse = "s",tipoArq = "jpg")
    else:
        baseAtual = imgsParaBase("Bases/%s"%base,qtClasses=qtCla)
    #EigenFace
    for j in range(0,41,5):
        erro = 0
        for i in range(hold):
            eigen = FractionalEigenfaces()
            train_atr, test_atr, train_classes, test_classes = train_test_split(baseAtual.atributos, baseAtual.classes, test_size=0.5, random_state=i) 
            bTreino = Base(train_classes,train_atr)
            bTeste = Base(test_classes,test_atr)
            eigen.fit(bTreino,r)
            if(j==0):
                j=1
            bTreino = eigen.run(bTreino,j)
            bTeste = eigen.run(bTeste,j)
            erro = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes,1) + erro
        resultados = "%s,%s\n"%(j,1 - (erro/hold))
        print(resultados)
        arqSave.write(resultados)
    arqSave.close()

