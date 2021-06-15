import numpy as np
import yaml
import os

from utils_v2 import get_data
from multilayer_perceptron import MultilayerPerceptron, Layer, load_mlp, save_mlp

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

if __name__ == "__main__": 
    
    epochs=5000
    learning_rate=0.0001
    font='font1'

    X, Y = get_data(font, fonts_training)

    greedy_layers = [
        'ej1a_gl30_2',
        'ej1a_gl25',
        'ej1a_gl20_2',
        'ej1a_gl15_2',
        'ej1a_gl10_4',
        'ej1a_gl2_2'
    ]

    greedy_models = [ load_mlp(name=model_name, dir=os.path.join(saves_folder, 'ej1a')) for model_name in greedy_layers ]

    layers = []
    for gm in greedy_models:
        layers.append(gm.layers[0])
    for gm in reversed(greedy_models):
        layers.append(gm.layers[-1])

    mlp = MultilayerPerceptron(layers)
    mae = mlp.calculate_mean_error(X, Y)
    print(f'MAE = {mae}')

    