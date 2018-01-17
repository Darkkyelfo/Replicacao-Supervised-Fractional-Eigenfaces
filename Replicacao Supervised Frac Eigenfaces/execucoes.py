'''
Created on 30 de dez de 2017

@author: raul1
'''
from classificadores import classicarKNN
from eigenfaces import Eigenfaces, FractionalEigenfaces, SupervisedFractionalEigenfaces
from imagemparabase import imgsParaBase
from pca import DualSupervisedPCA
from dividirbase import Holdout


def executarTodos(base,qtCla=15,r=0.01,hold=10):
    #Bases
    if(base=="georgia"):
        baseAtual = imgsParaBase("Bases/%s"%base,qtClasses=qtCla,dirClasse = "s",tipoArq = "jpg")
    else:
        baseAtual = imgsParaBase("Bases/%s"%base,qtClasses=qtCla)

    basesHold = gerarBases(baseAtual,hold+1)
    bTreinoOri = basesHold[0][0]

    #cria os redutores de dimensionalidade

    pca = DualSupervisedPCA()
    pca.fit(bTreinoOri)
    eigen = Eigenfaces()
    eigen.fit(bTreinoOri)
    eigenFrac = FractionalEigenfaces()
    eigenFrac.fit(bTreinoOri, r)
    supEigen = SupervisedFractionalEigenfaces()
    supEigen.fit(bTreinoOri, r)

    for j in range(1,41):
        erros = [0]*4
        print(j)
        for k in basesHold[1:]:
            bTesteOri = k[0]
            bTreinoOri = k[1]

            bTreino = pca.run(bTreinoOri,j)
            bTeste = pca.run(bTesteOri,j)
            erros[0] = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes,1) + erros[0]

            bTreino = eigen.run(bTreinoOri,j)
            bTeste = eigen.run(bTesteOri,j)
            erros[1] = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes, 1) + erros[1]

            bTreino = eigenFrac.run(bTreinoOri,j)
            bTeste = eigenFrac.run(bTesteOri,j)
            erros[2] = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes, 1) + erros[2]

            bTreino = supEigen.run(bTreinoOri,j)
            bTeste = supEigen.run(bTesteOri,j)
            erros[3] = classicarKNN(bTreino.atributos, bTreino.classes, bTeste.atributos, bTeste.classes, 1) + erros[3]

        for i,erro in enumerate(erros):
            if i == 0 :
                print("DSPCA_%s"%base + "\n%s,%s\n"%(j,erro/hold))
                criarCSV("DSPCA_%s"%base,"%s,%s\n"%(j,erro/hold))
            elif i== 1:
                print("EIGEN_%s" % base + "\n%s,%s\n" % (j, erro / hold))
                criarCSV("EIGEN_%s" % base, "%s,%s\n" % (j, erro / hold))
            elif i== 2:
                print("FRACEIGEN_%s" % base + "\n%s,%s\n" % (j, erro / hold))
                criarCSV("FRACEIGEN_%s" % base, "%s,%s\n" % (j, erro / hold))
            elif i == 3:
                print("SUPEIGEN_%s" % base + "\n%s,%s\n" % (j, erro / hold))
                criarCSV("SUPEIGEN_%s" % base, "%s,%s\n" % (j, erro / hold))

def gerarBases(baseAtual,hold):
    bases = []
    for k in range(hold):
        bTesteOri, bTreinoOri = Holdout.dividirImg(baseAtual)
        bases.append([bTesteOri,bTreinoOri])
    return bases

def criarCSV(nome,valores):
    caminhoArq = "Resultados/%s"%nome+".csv"
    arqSave = open(caminhoArq,"a")
    arqSave.write(valores)
    arqSave.close()

    

