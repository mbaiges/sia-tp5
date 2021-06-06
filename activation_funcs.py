import math
import numpy as np

# Sign

def sign_activation(excitement):
    if excitement > 0:
        return 1
    elif excitement < 0:
        return -1
    else:
        return 0

def dx_sign_activation(excitement):
    return 0
        
# Tanh
        
def tanh_activation(excitement):
    beta = 2	
    return np.tanh(beta*excitement)
    
def dx_tanh_activation(excitement):
    beta = 2
    return beta/np.cosh(beta*excitement)**2
    # return 1 - (tanh_activation(excitement)**2) 

# Lineal

def lineal_activation(excitement):
    return excitement

def dx_lineal_activation(excitement):
    return 1

# ReLU

def relu_activation(excitement):
    return max(0, excitement)

def dx_relu_activation(excitement):
    if excitement >= 0:
        return 1
    return 0

# Sigmoid

def sigmoid_activation(excitement):
    return 1/(1 + np.exp(-excitement))

def dx_sigmoid_activation(excitement):
    return sigmoid_activation(excitement) * (1 - sigmoid_activation(excitement))