'''
Created on 30 de dez de 2017

@author: raul1
'''
from execucoes import executarTodos
if __name__ == '__main__':
    
    r = 0.1
    hold = 50
    '''
    execucaoSupFracEigen("orl", "resultadoSupEigenOrl.txt",40,r,hold)
    execucaoEigen("orl", "resultadoEigenOrl.txt", 40, hold)
    execucaoFracEigen("orl", "resultadoFracEigenOrl.txt",40,r,hold)
    
    execucaoSupFracEigen("yale", "resultadoSupEigenYale.txt",15,r,50)
    execucaoEigen("yale", "resultadoEigenYale.txt", 15, hold)
    execucaoFracEigen("yale", "resultadoFracEigenYale.txt",15,r,hold)

    execucaoFracEigen("georgia", "resultadoFracEigenGeorgia.txt",50,r,hold)
    execucaoSupFracEigen("georgia", "resultadoSupEigenGeorgia.txt",50,r,hold)
    execucaoEigen("georgia", "resultadoEigenGeogia.txt",50, hold)
    '''
    #executarDSPCA("yale", "resultadoDSPCAYale.txt",15,50)
    executarTodos("yale",15,r,hold)
   # executarTodos("orl", "resultadoTodosOrl.txt",40,r,hold)
   # executarTodos("georgia", "resultadoTodosGeorgia.txt",50,r,hold)
    #executarDSPCA("orl", "resultadoDSPCAOrl.txt",40,50)
   # executarDSPCA("georgia", "resultadoDSPCAGeorgia.txt",50,50)
    
    pass