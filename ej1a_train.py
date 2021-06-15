import yaml
import os

from utils_v2 import get_data
from multilayer_perceptron import MultilayerPerceptron, Layer, save_mlp

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

    mlp = MultilayerPerceptron([
        Layer(neuron_units=20, activation='tanh', input_size=X.shape[1]),
        Layer(neuron_units=10, activation='tanh'),
        Layer(neuron_units=2, activation='tanh'),
        Layer(neuron_units=10, activation='tanh'),
        Layer(neuron_units=20, activation='tanh'),
        Layer(neuron_units=Y.shape[1], activation='tanh')
    ])

    mlp.init_weights()
    mlp.fit(X, Y, learning_rate=learning_rate, epochs=epochs)
    # save_mlp(mlp, name=f"ej1a_{font}_{epochs}", dir=os.path.join(saves_folder, 'ej1a'))
    save_mlp(mlp, name=f"ej1a_no_mom", dir=os.path.join(saves_folder, 'ej1a'))