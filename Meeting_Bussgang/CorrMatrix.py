# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:04:45 2024

@author: AM271697
"""
###### Importing modules
from scipy.io import savemat, loadmat
import numpy as np
import pandas as pd

###### Functions
def correlation_matrix(X, Y):
    # Calculate correlation coefficient between each element of X and elements in Y
    print(X.shape)
    corr_matrix = np.zeros((X.shape[1], Y.shape[1]))
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            corr_matrix[i, j] = np.corrcoef(X[:,i], Y[:,j])[0, 1]
    return corr_matrix

###### Parameters
L = 1
M = 100
K = 15
Mfft = 256
wp_dBm = -92
wp = 0.001*(10**(wp_dBm/10))

###### Correlationbetween Xin, Xout and Distortion
data1 = loadmat("./BussgangTheorem_K%d_L%d_M%d_MFFT%d.mat" % (K,L,M,Mfft))

Xin = data1["Xin"]
Xout = data1["Xout"]
Dis = data1["DisFreq"]


Nreal = Xin.shape[0]
Rout = np.zeros((Nreal,M,M),dtype=complex)
Rd = np.zeros((Nreal,M,M),dtype=complex)
Rtest = np.zeros((Nreal),dtype=complex)
for iNreal in range(Nreal):
    A = Xin[iNreal,:,0,0].reshape(M,1)
    B = Xout[iNreal,:,0,0].reshape(M,1)
    C = Dis[iNreal,:,0,0].reshape(M,1)
    Rout[iNreal,:,:] = A.dot(np.conjugate(B).T)
    Rd[iNreal,:,:] = A.dot(np.conjugate(C).T)
    Rtest[iNreal] =  Xin[iNreal,0,0,0]*np.conjugate(Dis[iNreal,3,0,0])
    
    
test_out = np.mean(Rout,axis=0)*wp/M
test_d = np.mean(Rd,axis=0)*wp/M
df = pd.DataFrame(test_out)
df.to_csv('pd.csv')
C = np.correlate(Xin[:,0,0,0],Dis[:,0,0,0])
# Generate correlation matrix
corr_matrix_xin_d = correlation_matrix(Xin[:,:,0,0],Dis[:,:,0,0])
corr_matrix_xin_xout = correlation_matrix(Xin[:,:,0,0],Xout[:,:,0,0])
###### Correlation between terms in SINR 
data2 = loadmat("./Terms_K%d_L%d_M%d_MFFT%d.mat" % (K,L,M,Mfft))

T1 = data2["T1"][0] # CPk
T2 = data2["T2"][0] # PUk
T3 = data2["T3"][0] # UIk
T4 = data2["T5"][0] # Dk
T5 = data2["T4"][0] # Nk

Termes = np.array([T1, T2, T3, T4, T5])

meanR = np.zeros((Nreal),dtype=complex)
for iNreal in range(Nreal):
    A = T1[iNreal]
    B = T3[iNreal]
    meanR[iNreal] =  A*np.conjugate(B)
    
    
reslt1 = np.mean(meanR,axis=0)


meanRR = np.zeros((Nreal,5,5),dtype=complex)
for iNreal in range(Nreal):
    for i in range(5):
        A = Termes[i,iNreal]
        for j in range(5):
            B = Termes[j,iNreal]
            meanRR[iNreal,i,j] =  A*np.conjugate(B)
    
    
reslt = np.mean(meanRR,axis=0)


