import numpy as np
from numpy.lib.npyio import save
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
    
    epochs=10000
    learning_rate=0.0001
    font='font1'

    X, Y = get_data(font, fonts_training)

    greedy_layers = [20, 10, 2]
    greedy_layers.insert(0, X.shape[1])

    greedy_models = []

    for i in range(1, len(greedy_layers)):
        outer_n = greedy_layers[i-1]
        inner_n = greedy_layers[i]
        mlp = MultilayerPerceptron([
            Layer(neuron_units=inner_n, activation='tanh', input_size=outer_n),
            Layer(neuron_units=outer_n, activation='tanh'),
        ])
        mlp.init_weights()
        greedy_models.append(mlp)

    last_X = X.copy()
    for i in range(len(greedy_models)):
        mlp = greedy_models[i]
        n = greedy_layers[i+1]
        mlp.fit(last_X, last_X, learning_rate=learning_rate, epochs=epochs)
        mae = mlp.min_err
        print(f'MAE at layer "{n}": {mae}')
        mlp_encoder = MultilayerPerceptron([
            mlp.layers[0]
        ])
        last_X = np.array([mlp_encoder.predict(x) for x in last_X])

    layers = []
    for gm in greedy_models:
        layers.append(gm.layers[0])
    for gm in reversed(greedy_models):
        layers.append(gm.layers[-1])

    mlp = MultilayerPerceptron(layers)
    mae = mlp.calculate_mean_error(X, Y)
    mlp.min_err = mae
    save_mlp(mlp, name='ej1a_gl', dir=os.path.join(saves_folder, 'ej1a'))
    print(f'MAE = {mae}')