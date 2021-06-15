from typing import final
import numpy as np
import math
import random
import yaml
import json
import os

from utils_v2 import progress_bar

show_progress_bar = False
momentum = False
momentum_mult = 0.8
adaptative_lr = False

config_filename = 'config.yaml'

with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    show_progress_bar = config['progress_bar']
    momentum = config['momentum']
    momentum_mult = config['momentum_mult']
    adaptative_lr = config['adaptative_lr']

class Layer:

    def __init__(self, neuron_units=0, activation='linear', input_size=None):
        self.neuron_units = neuron_units
        self.activation = activation
        self.fn, self.df = self._select_activation_fn(activation)
        self.input_size = input_size
        self.delta_weights = None
        self.delta_biases = None

    def _select_activation_fn(self, activation):
        if activation == 'relu':
            fn = lambda x: np.where(x < 0, 0.0, x)
            df = lambda x: np.where(x < 0, 0.0, 1.0)
        elif activation == 'sigmoid':
            fn = lambda x: 1 / (1 + np.exp(-x))
            df = lambda x: x * (1 - x)
        elif activation == 'tanh':
            fn = lambda x: np.tanh(2*x)
            df = lambda x: 2/np.cosh(2*x)**2
        elif activation == 'linear' or activation is None:
            fn = lambda x: x
            df = lambda x: 1.0
        else:
            NotImplementedError(f"Function {activation} cannot be used.")
        return fn, df

    def set_weights(self, W, B):
        self.W = W
        self.B = B

    def init_weights(self, weights_n=0):
        # xavier
        self.weights_n = weights_n
        self.W = np.random.normal(loc=0.0, size=(weights_n, self.neuron_units), scale=np.sqrt(2/(weights_n + self.neuron_units)))
        self.B = np.random.normal(loc=0.0, size=self.neuron_units, scale=np.sqrt(2/(weights_n + self.neuron_units)))


    def _forward(self, X):     
        # 1x4  4x3  = 1x3      N1   N2  N3
        # (x1, x2, x3 x4)  x  (p11 p21 p31)  = (a b c)
        #                     (p12 p22 p13)
        #                     (p13 p23 p33)
        #                     (p14 p24 p34)
        # print(X)
        # print(self.W)
        H = X @ self.W
        self.H = H + self.B
        A = self.fn(self.H)
        self.A = A
        return A
    
    def backpropagate(self, diff, prev_activation, learning_rate):

        delta = np.multiply(diff, self.df(self.H)) # mult punto a punto
        self.delta = delta

        delta = np.array([delta])
        prev_activation = np.array([prev_activation])
        
        aux = delta.T @ prev_activation
        
        delta_weights = learning_rate * aux.T
        delta_biases = learning_rate * delta[0]

        if momentum and not self.delta_weights is None and not self.delta_biases is None:
            delta_weights = delta_weights + momentum_mult * self.delta_weights
            delta_biases = delta_biases + momentum_mult * self.delta_biases

        self.W = self.W + delta_weights
        self.B = self.B + delta_biases

        self.delta_weights = delta_weights
        self.delta_biases = delta_biases

        new_diff = self.W @ delta.T

        return new_diff.T[0]

class MultilayerPerceptron:

    def __init__(self, layers=[]): #[5, 3, 5]
        self.layers = layers

    def init_weights(self):
        for i, layer in enumerate(self.layers):
            weights_n = layer.input_size if i == 0 else self.layers[i-1].neuron_units
            layer.init_weights(weights_n)

    def _forward(self, X):
        last = X
        for layer in self.layers:
            last = layer._forward(last)
        return last

    def _backpropagate(self, X, Y_pred, Y, learning_rate):
        diff = Y - Y_pred

        for idx in range(len(self.layers)-1, -1, -1): # itera desde arriba hacia abajo
            layer = self.layers[idx]
            if idx == 0:
                A = X
            else:
                A = self.layers[idx-1].A

            diff = layer.backpropagate(diff, A, learning_rate)

        # 1x4  4x3  = 1x3      N1   N2  N3
        # (x1, x2, x3 x4)  x  (p11 p21 p31)  = (a b c)
        #                     (p12 p22 p13)
        #                     (p13 p23 p33)
        #                     (p14 p24 p34)

        # 1x4  4x3  = 1x3      N1   N2  N3
        # (d1, d2, d3)  x     (V11, V12, V13, V14)  = (d1.v11 d2.v11 d3.v11)
        #                                           = (d1.v12 d2.v12 d3.v12)
        #                                           = (d1.v13 d2.v13 d3.v13)
        #                                           = (d1.v14 d2.v14 d3.v14)


    def _get_layers_cfg(self):
        return [{'neuron_units': layer.neuron_units, 'activation': layer.activation, 'W': layer.W.copy(), 'B': layer.B.copy()} for layer in self.layers]

    def predict(self, X):
        return self._forward(X)

    def fit(self, X, Y, learning_rate=0.001, epochs=100):
        lr0 = learning_rate
        lr = learning_rate
        limit = 2
        if adaptative_lr:
            final_lr = learning_rate/10
            d =  (-(epochs/limit)/math.log(final_lr/lr0))

        # X = self._transform_X(X)
        examples_n = X.shape[0]
        self.min_err = math.inf

        examples_order = [i for i in range(0, examples_n)]


        for epoch_n in range(epochs):

            if adaptative_lr and epoch_n < epochs/limit:
                lr = lr0 * math.exp((-1/d)*epoch_n)
            
            if show_progress_bar:
                progress_bar(epoch_n, epochs)

            random.shuffle(examples_order)
            
            for i in examples_order:
                X_example = X[i]
                Y_example = Y[i]

                Y_predicted = self._forward(X_example)
                self._backpropagate(X_example, Y_predicted, Y_example, lr)

                err = self.calculate_mean_error(X, Y)
                # print(f'Error: {err:.2f}')

                if err < self.min_err:
                    self.min_err = err
                    self.best_layers_cfg = self._get_layers_cfg()

        if show_progress_bar:
            progress_bar(epochs, epochs)
            print('\n\n', end='')

        # print(self._forward(X[0]))
        print(f'Error: {self.min_err:.2f}')

    def load_layers_cfg(self, layers_cfg):
        layers = []
        for cfg in layers_cfg:
            l = Layer(cfg['neuron_units'], cfg['activation'])
            l.set_weights(cfg['W'], cfg['B'])
            layers.append(l)
        self.layers = layers

    @property
    def best_model(self):
        return {
            'layers': self._get_layers_cfg(),
            'mae': self.min_err if not self.min_err is None else math.inf
        }
    
    def calculate_abs_error(self, X, Y):
        examples_n = X.shape[0]
        err = 0
        for i in range(0, examples_n):
            err += sum( abs( self._forward(X[i]) - Y[i] ) )

        return err

    def calculate_mean_error(self, X, Y):
        err = self.calculate_abs_error(X, Y)
        err /= X.shape[0]
        return err

def save_mlp(mlp, name='', dir=''):
    model = mlp.best_model

    data = {
        'name': name,
        'layers': model['layers'],
        'mae': model['mae']
    }

    # np.array to list
    for l in data['layers']:
        l['W'] = l['W'].tolist()
        l['B'] = l['B'].tolist()

    if dir != '' and (not os.path.exists(dir) or not os.path.isdir(dir)):
        os.mkdir(dir)

    if os.path.exists(os.path.join(dir, name) + '.json'):
        base_name = name
        idx = 2
        exists = True
        while exists:
            exists = False
            name = f'{base_name}_{idx}'
            if os.path.exists(os.path.join(dir, name) + '.json'):
                exists = True
            idx += 1

    name += '.json'

    full_path = os.path.join(dir, name)

    with open(full_path, 'w') as outfile:
        json.dump(data, outfile)

    s = f'SAVE: Multilayer Perceptron named "{name}" saved'
    if dir != '.':
        s += f' in directory "{dir}"'
    print(f'\n{s}\n')

def load_mlp(name='', dir=''):
    name += '.json'

    full_path = os.path.join(dir, name)
    if not os.path.exists(full_path):
        print(f'ERROR: Load failed. File "{name}" does not exist')
        exit(1)

    with open(full_path) as json_file:
        data = json.load(json_file)

        # list to np.array
        for l in data['layers']:
            l['W'] = np.array(l['W'])
            l['B'] = np.array(l['B'])

        mlp = MultilayerPerceptron()
        mlp.load_layers_cfg(data['layers'])

        s = f'LOAD: Loaded Multilayer Perceptron named {name} (mae={data["mae"]:.2f})'
        print(f'\n{s}\n')
        return mlp