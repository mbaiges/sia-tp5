import numpy as np
import yaml
import os

from utils_v2 import get_data
from multilayer_perceptron import MultilayerPerceptron, Layer, load_mlp

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

    mlp_gl30 = MultilayerPerceptron([
        Layer(neuron_units=30, activation='tanh', input_size=X.shape[1]),
        Layer(neuron_units=Y.shape[1], activation='tanh')
    ])
    mlp_gl30.init_weights()

    mlp_gl25 = MultilayerPerceptron([
        Layer(neuron_units=25, activation='tanh', input_size=30),
        Layer(neuron_units=30, activation='tanh')
    ])
    mlp_gl25.init_weights()

    mlp_gl20 = MultilayerPerceptron([
        Layer(neuron_units=20, activation='tanh', input_size=25),
        Layer(neuron_units=25, activation='tanh')
    ])
    mlp_gl20.init_weights()

    mlp_gl15 = MultilayerPerceptron([
        Layer(neuron_units=15, activation='tanh', input_size=20),
        Layer(neuron_units=20, activation='tanh')
    ])
    mlp_gl15.init_weights()

    mlp_gl10 = MultilayerPerceptron([
        Layer(neuron_units=10, activation='tanh', input_size=15),
        Layer(neuron_units=15, activation='tanh')
    ])
    mlp_gl10.init_weights()

    mlp_gl2 = MultilayerPerceptron([
        Layer(neuron_units=2, activation='tanh', input_size=10),
        Layer(neuron_units=10, activation='tanh')
    ])
    mlp_gl2.init_weights()

    last_X = X.copy()

    # train 30 layer
    # mlp_gl30.fit(last_X, last_X, learning_rate=learning_rate, epochs=epochs)
    # print(f'Error: {mlp_gl30.min_err}')
    # save_mlp(mlp_gl30, name=f"ej1a_gl30", dir=os.path.join(saves_folder, 'ej1a'))

    # generate X
    mlp_gl30 = load_mlp(name='ej1a_gl30_2', dir=os.path.join(saves_folder, 'ej1a'))

    mlp_gl30_encoder = MultilayerPerceptron([
        mlp_gl30.layers[0]
    ])

    new_X = []
    for i in range(last_X.shape[0]):
        new_X.append(mlp_gl30_encoder.predict(last_X[i]))
    last_X = np.array(new_X)
    print(last_X.shape)

    # train 25 layer
    # mlp_gl25.fit(last_X, last_X, learning_rate=learning_rate, epochs=epochs)
    # print(f'Error: {mlp_gl25.min_err}')
    # save_mlp(mlp_gl25, name=f"ej1a_gl25", dir=os.path.join(saves_folder, 'ej1a'))

    # generate X
    mlp_gl25 = load_mlp(name='ej1a_gl25', dir=os.path.join(saves_folder, 'ej1a'))

    mlp_gl25_encoder = MultilayerPerceptron([
        mlp_gl25.layers[0]
    ])

    new_X = []
    for i in range(last_X.shape[0]):
        new_X.append(mlp_gl25_encoder.predict(last_X[i]))
    last_X = np.array(new_X)
    print(last_X.shape)

    # train 20 layer
    # mlp_gl20.fit(last_X, last_X, learning_rate=learning_rate, epochs=epochs)
    # print(f'Error: {mlp_gl20.min_err}')
    # save_mlp(mlp_gl20, name=f"ej1a_gl20", dir=os.path.join(saves_folder, 'ej1a'))

    # generate X
    mlp_gl20 = load_mlp(name='ej1a_gl20_2', dir=os.path.join(saves_folder, 'ej1a'))

    mlp_gl20_encoder = MultilayerPerceptron([
        mlp_gl20.layers[0]
    ])

    new_X = []
    for i in range(last_X.shape[0]):
        new_X.append(mlp_gl20_encoder.predict(last_X[i]))
    last_X = np.array(new_X)
    print(last_X.shape)

    # train 15 layer
    # mlp_gl15.fit(last_X, last_X, learning_rate=learning_rate, epochs=epochs)
    # print(f'Error: {mlp_gl15.min_err}')
    # save_mlp(mlp_gl15, name=f"ej1a_gl15", dir=os.path.join(saves_folder, 'ej1a'))

    # generate X
    mlp_gl15 = load_mlp(name='ej1a_gl15_2', dir=os.path.join(saves_folder, 'ej1a'))

    mlp_gl15_encoder = MultilayerPerceptron([
        mlp_gl15.layers[0]
    ])

    new_X = []
    for i in range(last_X.shape[0]):
        new_X.append(mlp_gl15_encoder.predict(last_X[i]))
    last_X = np.array(new_X)
    print(last_X.shape)

    # train 10 layer
    # mlp_gl10.fit(last_X, last_X, learning_rate=learning_rate, epochs=epochs)
    # print(f'Error: {mlp_gl10.min_err}')
    # save_mlp(mlp_gl10, name=f"ej1a_gl10", dir=os.path.join(saves_folder, 'ej1a'))

    # generate X
    mlp_gl10 = load_mlp(name='ej1a_gl10_4', dir=os.path.join(saves_folder, 'ej1a'))

    mlp_gl10_encoder = MultilayerPerceptron([
        mlp_gl10.layers[0]
    ])

    new_X = []
    for i in range(last_X.shape[0]):
        new_X.append(mlp_gl10_encoder.predict(last_X[i]))
    last_X = np.array(new_X)
    print(last_X.shape)

    # train 2 layer
    # mlp_gl2.fit(last_X, last_X, learning_rate=learning_rate, epochs=epochs)
    # print(f'Error: {mlp_gl2.min_err}')
    # save_mlp(mlp_gl2, name=f"mlp_gl2", dir=os.path.join(saves_folder, 'ej1a'))
