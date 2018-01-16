import random
from copy import deepcopy
from base import BaseImg
from numba import jit

class Holdout(object):

    @staticmethod
    @jit
    def dividirImg(base, porc=0.5, estratificada=True):
        baseTeste = base.copy()
        baseTesteQtPclasse = deepcopy(baseTeste.qtPorClasse)
        if (estratificada):
            atributos = []
            classes = []
            mImages = []
            while Holdout._verificarQtClasse(classes,porc,baseTesteQtPclasse):
                ind = random.randint(0, baseTeste.qtElementos-1)
                classe = baseTeste.classes[ind]
                if classes.count(classe) < int(porc * baseTesteQtPclasse[classe]):
                    classes.append(classe)
                    atributos.append(baseTeste.atributos[ind])
                    mImages.append(baseTeste.matrizImgs[ind])
                    baseTeste.remover(ind)
        baseTreino = BaseImg(classes, atributos, mImages)
        return baseTeste,baseTreino

    @staticmethod
    def _verificarQtClasse(classes,porc,qtPorClasseList):
        for i in qtPorClasseList:
            if classes.count(i) < int(porc * qtPorClasseList[i]):
                return True
        return False

