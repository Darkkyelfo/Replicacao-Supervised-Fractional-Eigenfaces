'''
Created on 30 de dez de 2017

@author: raul1
'''
from execucoes import execucaoEigen, execucaoSupFracEigen,execucaoFracEigen
if __name__ == '__main__':
    
    r = 0.1
    hold = 50
    #execucaoEigen("yale", "resultadoSupEigenYale.txt",15,r,50)
    #execucaoSupFracEigen("orl", "resultadoSupEigenOrl.txt",40,r,hold)
    execucaoSupFracEigen("georgia", "resultadoSupEigenGeorgia.txt",50,r,hold)
    
    execucaoEigen("yale", "resultadoEigenYale.txt", 15, hold)
    execucaoEigen("orl", "resultadoEigenOrl.txt", 40, hold)
    execucaoEigen("georgia", "resultadoEigenGeogia.txt",50, hold)
    
    execucaoFracEigen("yale", "resultadoFracEigenYale.txt",15,r,hold)
    execucaoFracEigen("orl", "resultadoFracEigenOrl.txt",40,r,hold)
    execucaoFracEigen("georgia", "resultadoFracEigenGeorgia.txt",50,r,hold)
    
    pass