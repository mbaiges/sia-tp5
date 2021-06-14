import numpy as np
import json

def backline():        
    print('\r', end='') # use '\r' to go back

def progress_bar(epoch_n, epochs):
    if epoch_n > 0:
        backline()
    pctg = ((epoch_n/epochs)*100)
    s = 'Progress: '
    s += '['
    inc = 5
    for p in range(0, 100, inc):
        if p < pctg:
            s += '='
        elif p - inc < pctg:
            s += '>'
        else:
            s += ' '
    s += f'] ({int(pctg)}%)'
    print(s, end='')

def get_data(font, fonts_file):
    
    f = open(fonts_file)
    data = json.loads(f.read())

    data1 = data[font]

    def plain_data(matrix):
        arr = []
        for row in matrix:
            for el in row:
                arr.append(el)
        return arr

    X = []

    for el in data1:
        X.append(plain_data(el))

    X = np.array(X)
    X = 2*X - 1

    return X, X
