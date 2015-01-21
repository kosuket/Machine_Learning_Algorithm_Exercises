# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Demonstration of the Perceptron and Linear Regressor on parity problem

import numpy as np
inputs = np.array([[0,0,0,0],[0,1,0,0],[1,0,0,0],[1,1,0,0],[0,0,1,0],[0,1,1,0],[1,1,1,0],[0,0,1,1],[0,1,1,1],[1,1,1,1]])
# PAR data
PARtargets = np.array([[1],[0],[0],[1],[0],[1],[0],[1],[0],[1]])

import pcn_logic_eg
import linreg

print "PAR logic function"
pPAR = pcn_logic_eg.pcn(inputs,PARtargets)
pPAR.pcntrain(inputs,PARtargets,0.25,6)


testin = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
# Parity
print "PAR data regress"
PARbeta = linreg.linreg(inputs,PARtargets)
PARout = np.dot(testin,PARbeta)
print PARout