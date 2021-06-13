import numpy as np
import math
import random

from utils_v2 import progress_bar

class Neuron:

    def __init__(self, fn, df):
        self.fn = fn
        self.df = df

        self.weights = None
        self.last_delta = None
        self.last_h = None
        self.last_a = None

    def init_random_weights_and_bias(self, weights_n, neuron_units):
        # xavier
        self.weights = np.random.normal(loc=0.0, size=weights_n, scale=np.sqrt(2/(weights_n + neuron_units)))
        self.bias = np.random.normal(loc=0.0, size=1, scale=np.sqrt(2/(weights_n + neuron_units)))[0]

    def eval(self, X): # h = Excitación |  a = Activación
        h = np.inner(self.weights, X) + self.bias
        self.last_h = h
        a = self.fn(h)
        self.last_a = a
        return a

    def apply_correction(self, entry, diff, learning_rate):
        delta = diff * self.df(self.last_h)
        self.last_delta = delta
        self.weights = self.weights + learning_rate * delta * entry
        self.bias = self.bias + learning_rate * delta

class Layer:

    def __init__(self, neuron_units=0, activation='linear', input_size=None):
        self.neuron_units = neuron_units
        self.fn, self.df = self._select_activation_fn(activation)
        self.input_size = input_size
        self.neurons = [Neuron(self.fn, self.df) for i in range(neuron_units)]

    def _select_activation_fn(self, activation):
        if activation == 'relu':
            fn = lambda x: np.where(x < 0, 0.0, x)
            df = lambda x: np.where(x < 0, 0.0, 1.0)
        elif activation == 'sigmoid':
            fn = lambda x: 1 / (1 + np.exp(-np.clip(x,-500,500)))
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

    def init_weights(self, weights_n=0):
        self.weights_n = weights_n
        for n in self.neurons:
            n.init_random_weights_and_bias(weights_n, self.neuron_units)

    def _forward(self, X):     
        res = np.array([n.eval(X) for n in self.neurons])
        self.A = res
        return res
    
    def backpropagate(self, diff, prev_activation, learning_rate):
        for n_idx in range(0, self.neuron_units):
            n = self.neurons[n_idx]
            n.apply_correction(prev_activation, diff[n_idx], learning_rate)

        new_diff = np.zeros(self.weights_n)
        for w_idx in range(0, self.weights_n):
            for n_idx in range(0, self.neuron_units):
                n = self.neurons[n_idx]
                new_diff[w_idx] += n.weights[w_idx] + n.last_delta
    
        return new_diff

class MultilayerPerceptron:

    def __init__(self, layers): #[5, 3, 5]
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

    def predict(self, X):
        return self._forward(X)

    def fit(self, X, Y, learning_rate=0.001, epochs=100):

        # X = self._transform_X(X)
        examples_n = X.shape[0]
        min_err = math.inf

        examples_order = [i for i in range(0, examples_n)]

        for epoch_n in range(epochs):

            progress_bar(epoch_n, epochs)

            random.shuffle(examples_order)
            
            for i in examples_order:
                X_example = X[i]
                Y_example = Y[i]

                Y_predicted = self._forward(X_example)
                self._backpropagate(X_example, Y_predicted, Y_example, learning_rate)

                err = self.calculate_mean_error(X, Y)
                # print(f'Error: {err:.2f}')

                if err < min_err:
                    min_err = err

        progress_bar(epochs, epochs)
        print('\n\n', end='')

        # print(self._forward(X[0]))
        print(f'Error: {min_err:.2f}')
    
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