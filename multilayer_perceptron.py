import numpy as np
import math
import random

from utils_v2 import progress_bar

# class Neuron:

#     def __init__(self, fn, df):
#         self.fn = fn
#         self.df = df

#         self.W = None
#         self.last_delta = None
#         self.last_h = None
#         self.last_a = None

#     def init_random_weights(self, neuron_units):
#         self.W = np.random.rand(neuron_units)

#     def eval(self, X): # h = Excitación |  a = Activación
#         h = np.inner(self.W, X) 
#         self.last_h = h
#         a = self.fn(h)
#         self.last_a = a
#         return a

#     def apply_correction(self, entry, diff, learning_rate):
#         delta = diff * self.df(self.last_excitation)
#         self.last_delta = delta
#         self.W = self.W + learning_rate * delta * entry

class Layer:

    def __init__(self, neuron_units=0, activation='linear', input_size=None):
        self.neuron_units = neuron_units
        self.fn, self.df = self._select_activation_fn(activation)
        self.input_size = input_size

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

    def init_weights(self, weights_n=0):
        # self.W = np.random.rand(weights_n, self.neuron_units) # weights_n
        # self.B = np.random.rand(self.neuron_units) # biases
        
        # self.neurons = []
        # for i in range(len(neuron_units)):
        #     n = Neuron(fn, df)
        #     n.init_random_weights(neuron_units)
        #     self.neurons.append(n)

        # xavier

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
        # print("backpropagate ----------------------------------")

        # print("diff:---------------------------------------")
        # print(diff.shape)
        # print(diff)

        # print("self.df(self.H):---------------------------------------")
        # print(self.df(self.H).shape)
        # print(self.df(self.H))

        delta = np.multiply(diff, self.df(self.H)) # mult punto a punto
        self.delta = delta

        delta = np.array([delta])
        prev_activation = np.array([prev_activation])

        # print("delta:---------------------------------------")
        # print(delta.shape)
        # print(delta)

        # print("prev_activation:---------------------------------------")
        # print(prev_activation.shape)
        # print(prev_activation)
        
        aux = delta.T @ prev_activation

        # print("aux:---------------------------------------")
        # print(aux.shape)
        # print(aux)
        
        delta_weights = learning_rate * aux.T

        # print("delta_weights:---------------------------------------")
        # print(delta_weights.shape)
        # print(delta_weights)

        # print("self.W:---------------------------------------")
        # print(self.W.shape)
        # print(self.W)

        self.W = self.W + delta_weights
        
        # aux2 = diff.T @ np.identity(prev_activation.shape)

        delta_biases = learning_rate * delta[0]
        self.B = self.B + delta_biases

        new_diff = self.W @ delta.T
        # print(new_diff)
        # if math.isnan(new_diff):
        #     print("error")
        #     exit(1)

        # print("new_diff:---------------------------------------")
        # print(new_diff.shape)
        # print(new_diff)

        # print("new_diff.T:---------------------------------------")
        # print(new_diff.T.shape)
        # print(new_diff.T)
    
        return new_diff.T[0]

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


        
                
    # def _transform_X(self, X):
    #     to_append = np.ones((1, X.shape[0]))
    #     new_X = X.T
    #     new_X = np.append(to_append, new_X, axis=0)
    #     return new_X.T

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
