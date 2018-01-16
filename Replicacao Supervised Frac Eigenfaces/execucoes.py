'''
Created on 30 de dez de 2017

@author: raul1
'''
from classificadores import classicarKNN
from imagemparabase import imgsParaBase
from sklearn.model_selection import train_test_split
from eigenfaces import SupervisedFractionalEigenfaces,Eigenfaces,FractionalEigenfaces
from pca import DualSupervisedPCA
from base import Base
from dividirbase import Holdout

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
        resultados = "erro:%s,%s\n"%(j,(erro/hold))
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
        resultados = "erro: %s,%s\n"%(j,(erro/hold))
        print(resultados)
        arqSave.write(resultados)
    arqSave.close()

def executarDSPCA(base,arq,qtCla=15,hold=10):
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
            pca = DualSupervisedPCA()
            train_atr, test_atr, train_classes, test_classes = train_test_split(baseAtual.atributos, baseAtual.classes, test_size=0.5, random_state=i)
            bTreino = Base(train_classes,train_atr)
            bTeste = Base(test_classes,test_atr)
            pca.fit(bTreino)
            if(j==0):
                j=1
            bTreino = pca.run(bTreino,j)
            bTeste = pca.run(bTeste,j)
            erro = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes,1) + erro
        resultados = "erro:%s,%s\n"%(j,(erro/hold))
        print(resultados)
        arqSave.write(resultados)
    arqSave.close()


def executarTodos(base,arq,qtCla=15,r=0.01,hold=10):
    caminhoArq = "Resultados/%s"%arq
    arqSave = open(caminhoArq,"w")
    resultados = "atr,acerto\n"
    arqSave.write(resultados)
    arqSave.close()
    #Bases
    if(base=="georgia"):
        baseAtual = imgsParaBase("Bases/%s"%base,qtClasses=qtCla,dirClasse = "s",tipoArq = "jpg")
    else:
        baseAtual = imgsParaBase("Bases/%s"%base,qtClasses=qtCla)

    basesHold = gerarBases(baseAtual,hold)
    for j in range(15,41):
        arqSave = open(caminhoArq, "a")
        erros = [0]*4
        print(j)
        for k in basesHold:
            bTesteOri = k[0]
            bTreinoOri = k[1]

            pca = DualSupervisedPCA()
            pca.fit(bTreinoOri)
            bTreino = pca.run(bTreinoOri,j)
            bTeste = pca.run(bTesteOri,j)
            erros[0] = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes,1) + erros[0]

            # eigen = Eigenfaces()
            # eigen.fit(bTreinoOri)
            # bTreino = eigen.run(bTreinoOri,j)
            # bTeste = eigen.run(bTesteOri,j)
            # erros[1] = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes, 1) + erros[1]
            #
            # eigenFrac = FractionalEigenfaces()
            # eigenFrac.fit(bTreinoOri,r)
            # bTreino = eigenFrac.run(bTreinoOri,j)
            # bTeste = eigenFrac.run(bTesteOri,j)
            # erros[2] = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes, 1) + erros[2]
            #
            # supEigen = SupervisedFractionalEigenfaces()
            # supEigen.fit(bTreinoOri,r)
            # bTreino = supEigen.run(bTreinoOri,j)
            # bTeste = supEigen.run(bTesteOri,j)
            # erros[3] = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes, 1) + erros[3]

        for erro in erros:
            print(erro/hold)
        for ind,erro in enumerate(erros):
            if ind==0:
                arqSave.write("\nDSPCA %s \n erro:%s"%(j,erro/hold))
            elif ind == 1:
                arqSave.write("\nEigenfaces %s \n erro:%s" % (j, erro/hold))
            elif ind == 2:
                arqSave.write("\nFracEigen %s \n erro:%s" % (j, erro/hold))
            elif ind == 3:
                arqSave.write("\nSuperEigen %s \n erro:%s" % (j, erro/hold))
        arqSave.close()




def gerarBases(baseAtual,hold):
    bases = []
    for k in range(hold):
        bTesteOri, bTreinoOri = Holdout.dividirImg(baseAtual)
        bases.append((bTesteOri,bTreinoOri))
    return bases

    

