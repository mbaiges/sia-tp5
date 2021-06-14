import numpy as np
import yaml
import os
import json
import math
from matplotlib import pyplot as plt
import random
from multilayer_perceptron import MultilayerPerceptron, Layer, save_mlp, load_mlp

# Ej1a

## data filenames
fonts_training = ''

## saves folder
saves_folder = ''

config_filename = 'config.yaml'

with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    data_folder = config['data_folder']
    fonts_training = os.path.join(data_folder, config['fonts_training'])
    saves_folder = config['saves_folder']

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

    return X, X

def show_latent_scatter(encoder, X):
    fig, ax = plt.subplots()

    labels = ["_", "!", "\"", "#", "$", "%", "&", "\'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?"]
    x = [] # Puntos del espacio latente
    y = []
    for i in range(0, X.shape[0]):
        point = encoder.predict(X[i])
        print(f"point = {point}")
        x.append(point[0])
        y.append(point[1])
        print(f"p:({x[i]},{y[i]})")

    ax.scatter(x,y)

    for i in range(X.shape[0]):
        ax.annotate(labels[i], (x[i], y[i]))

    plt.show() 

if __name__ == "__main__": 
    X, Y = get_data()
    mlp = load_mlp(name="ej1a_1000", dir=saves_folder)
    encoder = MultilayerPerceptron(mlp.layers[:4])
    show_latent_scatter(encoder, X)
