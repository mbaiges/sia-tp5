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

if __name__ == "__main__":
    learning_rate=0.0001
    epochs=200000
    font='font1'

    X, Y = get_data(font, fonts_training)

    # mlp = load_mlp(name="ej1a_{font}_{epochs}", dir=os.path.join(saves_folder, 'ej1a'))
    mlp = load_mlp(name=f"ej1a_id7", dir=os.path.join(saves_folder, 'ej1a'))
    decoder = MultilayerPerceptron(mlp.layers[4:])
    num1 = 0
    num2 = 1
    character = decoder.predict([num1, num2]).reshape((7,5))
    plt.imshow(character)
    plt.show()