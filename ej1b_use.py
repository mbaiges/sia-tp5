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

def noisify(X, noise_coverage, noise_pct):
    new_X = []
    for i in range(0, X.shape[0]):
        new_X.append([])
        for j in range(0, X.shape[1]):
            aux = X[i][j] 
            must_apply = random.uniform(0, 1)
            if(must_apply < noise_coverage):
                noise = random.uniform(0, noise_pct)
                multiplyer = 1 if X[i][j] < 0 else -1
                aux = X[i][j] + noise * multiplyer
            new_X[i].append(aux)
    return np.array(new_X)

def show_noise_character_comparison(X_noise, Y, mlp):
    images = []

    for i in range(0, Y.shape[0]):
        images.append(Y[i].reshape((7,5)))

    for i in range(0, X.shape[0]):
        images.append(X_noise[i].reshape((7,5)))

    # for i in range(0, X.shape[0]):
    #     images.append(mlp.predict(X[i]).reshape((7,5)))
    for i in range(0, X.shape[0]):
        images.append(mlp.predict(X[i]).reshape((7,5)))

    fig, axes = plt.subplots(3, X.shape[0], figsize=(7,5))
    for i, ax in enumerate(axes.flat):
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.imshow(images[i])
    
    axes[0, 15].set_title("Expected Output",fontsize=20)
    axes[1, 15].set_title("Input With Noise",fontsize=20)
    axes[2, 15].set_title("Output",fontsize=20)

    plt.suptitle(f"Epocs: {epochs} - Learning Rate: {learning_rate}",fontsize=32)

    plt.show()

if __name__ == "__main__": 
    learning_rate=0.0001
    epochs=25000
    noise_coverage = 1
    noise_pct = 0.5

    X, Y = get_data()
    X_noise = noisify(X, noise_coverage, noise_pct)
    mlp = load_mlp(name=f"ej1b_{epochs}_{noise_coverage*100}_{noise_pct*100}_big", dir=saves_folder)
    show_noise_character_comparison(X_noise,Y,mlp)
