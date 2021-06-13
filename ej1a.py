import numpy as np
import yaml
import os
import json
import math
from multilayer_perceptron import MultilayerPerceptron, Layer

# Ej1

## data filenames
fonts_training = ''

config_filename = 'config.yaml'

with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    data_folder = config['data_folder']
    fonts_training = os.path.join(data_folder, config['fonts_training'])

def get_data():
    
    f = open(fonts_training)
    data = json.loads(f.read())

    data1 = data['font1']

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

    print(X.shape)
    print(X)

    return X, X

if __name__ == "__main__": 

    X, Y = get_data()

    # X = np.array([
    #     [-1, 1],
    #     [1, -1],
    #     [-1, -1],
    #     [1, 1]
    # ])

    # Y = np.array([[-1], [-1], [-1], [1]])

    # X, Y = np.array([[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 4, 5]])

    mlp = MultilayerPerceptron([
        Layer(neuron_units=20, activation='tanh', input_size=X.shape[1]),
        Layer(neuron_units=4, activation='tanh'),
        Layer(neuron_units=20, activation='tanh'),
        Layer(neuron_units=Y.shape[1], activation='tanh')   
    ])

    mlp.init_weights()
    mlp.fit(X, Y, learning_rate=0.0001, epochs=10000)

    for i in range(X.shape[0]):
        print(f'with x={X[i]} expected={Y[i]}, but got={mlp.predict(X[i])}')