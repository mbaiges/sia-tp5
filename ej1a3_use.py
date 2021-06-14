import yaml
import os
from matplotlib import pyplot as plt

from utils_v2 import get_data
from multilayer_perceptron import MultilayerPerceptron, load_mlp

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
    learning_rate=0.0001
    epochs=500
    font='font1'

    X, Y = get_data(font, fonts_training)

    mlp = load_mlp(name="ej1a_{font}_{epochs}", dir=os.path.join(saves_folder, 'ej1a'))
    encoder = MultilayerPerceptron(mlp.layers[:4])
    show_latent_scatter(encoder, X)
