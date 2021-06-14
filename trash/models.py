import numpy as np
import math
import random
import yaml
import os
import multiprocessing as mp
import keyboard

from utils import calculate_abs_error, calculate_mean_error
from plotters import plot_avg_error

plotting = False

config_filename = 'config.yaml'

# extra modes

with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    if 'plotting' in config:
        plotting = config['plotting']

class AutoEncoder:

    def __init__(self, X_shape, Z_shape, hidden_layers1, learning_level1, activation_func1, dx_activation_func1, hidden_layers2, learning_level2, activation_func2, dx_activation_func2):
        self.X_shape = X_shape
        self.Z_shape = Z_shape
        self.hidden_layers1 = hidden_layers1
        self.learning_level1 = learning_level1
        self.activation_func1 = activation_func1
        self.dx_activation_func1 = dx_activation_func1
        self.hidden_layers2 = hidden_layers2
        self.learning_level2 = learning_level2
        self.activation_func2 = activation_func2
        self.dx_activation_func2 = dx_activation_func2

        self.initialize()
        self.min_weights_and_bias1 = self.neural_network1.get_weights_and_bias()
        self.min_weights_and_bias2 = self.neural_network2.get_weights_and_bias()
        self.min_error = math.inf

    def initialize(self):
        self.neural_network1 = NeuralNetwork(self.hidden_layers1, self.X_shape, self.Z_shape, self.activation_func1, self.dx_activation_func1)
        self.neural_network2 = NeuralNetwork(self.hidden_layers2, self.Z_shape, self.X_shape, self.activation_func2, self.dx_activation_func2)

    def train(self, X, Y, epsilon=0, epochs=100, max_it_same_bias=1000):
        i = 0
        n = 0
        error = self.min_error

        #creat e random= index array 
        orders = [a for a in range(0, X.shape[0])]
        
        epoch_n = 0


        if plotting:
            plotter_q = mp.Queue()
            plotter_q.cancel_join_thread()

            plotter = mp.Process(target=plot_avg_error, args=((plotter_q),))
            plotter.daemon = True
            plotter.start()

        while epoch_n < epochs and error > epsilon:
            random.shuffle(orders)
            
            i = 0
            n = 0
            
            while i < len(orders) and error > epsilon:
            
                #check if perceptron needs resetting
                if n > max_it_same_bias * X.shape[0]:
                    self.initialize()
                    n = 0

                #choose random input
                rand_idx = np.random.randint(0, X.shape[0])
                rand_X = X[rand_idx, :]

                #evaluate chosen input
                activation1 = self.neural_network1.evaluate(rand_X)
                activation2 = self.neural_network2.evaluate(activation1)

                last_correction = self.neural_network2.apply_correction(self.learning_level2, rand_X-activation2, activation1)
                self.neural_network1.apply_correction(self.learning_level1, last_correction, rand_X)

                #calculate training error
                error = calculate_abs_error([self.neural_network1, self.neural_network2], X, X)

                if plotting:
                    mean_error = calculate_mean_error([self.neural_network1, self.neural_network2], X, X)
                    plotter_q.put({
                        'mean_error': mean_error
                    })

                if error < self.min_error:
                    self.min_error = error
                    self.min_weights_and_bias1 = self.neural_network1.get_weights_and_bias()
                    self.min_weights_and_bias2 = self.neural_network2.get_weights_and_bias()
                    #print('updated min_error', self.min_error)

                i += 1
                n += 1

            epoch_n += 1

        if plotting:
            plotter_q.put("STOP")
            print("Press 'q' to finish plot")
            keyboard.wait("q")

        return epoch_n >= epochs

    def get_best_model(self):
        return [NeuralNetwork(self.min_weights_and_bias1, self.hidden_layers1, self.X_shape, self.Z_shape, self.activation_func1, self.dx_activation_func1),NeuralNetwork(self.min_weights_and_bias2, self.hidden_layers2, self.Z_shape, self.X_shape, self.activation_func2, self.dx_activation_func2)]

class NetworkNeuron:

    def __init__(self, weights, bias, activation_func):
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func
        
    def apply_correction(self, learning_level, entry, delta): 
        self.last_delta = delta
        correction = learning_level * delta
        # print("NN-----------Correction:", correction)
        # print("NN-----------Entry:", entry)
        delta_weights = correction * entry
        delta_bias = correction
        self.weights += delta_weights
        self.bias += delta_bias
        
    def evaluate(self, entry):
        excitation = np.inner(entry, self.weights)
        self.last_excitation = excitation
        activation = self.activation_func(excitation + self.bias)
        self.last_activation = activation
        return activation

class MultilayerPerceptron:

    def __init__(self, hidden_layers, X_shape, Y_shape, learning_level, activation_func, dx_activation_func):
        self.hidden_layers = hidden_layers
        self.X_shape = X_shape
        self.Y_shape = Y_shape
        self.learning_level = learning_level
        self.activation_func = activation_func
        self.dx_activation_func = dx_activation_func
        self.initialize()
        self.min_weights_and_bias = self.neural_network.get_weights_and_bias()
        self.min_error = math.inf

    def initialize(self):
        self.neural_network = NeuralNetwork(self.hidden_layers, self.X_shape, self.Y_shape, self.activation_func, self.dx_activation_func)

    def train(self, X, Y, epsilon=0, epochs=100, max_it_same_bias=1000):
        i = 0
        n = 0
        error = self.min_error

        #creat e random= index array 
        orders = [a for a in range(0, X.shape[0])]
        
        epoch_n = 0

        if plotting:
            plotter_q = mp.Queue()
            plotter_q.cancel_join_thread()

            plotter = mp.Process(target=plot_avg_error, args=((plotter_q),))
            plotter.daemon = True
            plotter.start()

        while epoch_n < epochs and error > epsilon:

            print(f'Epoch percentage: {((epoch_n/epochs)*100):.2f}%')

            random.shuffle(orders)
            
            i = 0
            n = 0
            
            while i < len(orders) and error > epsilon:
            
                #check if perceptron needs resetting
                # if n > max_it_same_bias * X.shape[0]:
                #     self.initialize()
                #     n = 0

                #choose random input            
                rand_idx = np.random.randint(0, X.shape[0])
                rand_X = X[rand_idx, :]
                rand_Y = Y[rand_idx]

                #print("holaaa1")
                #evaluate chosen input
                # print("--------- Entry ---------")
                # print(rand_X)
                activation = self.neural_network.evaluate(rand_X)
                # print("--------- Output ---------")
                # print(activation)
                self.neural_network.apply_correction(self.learning_level, rand_Y-activation, rand_X)
                # print("holaaa2")
                #calculate training error
                error = calculate_abs_error(self.neural_network, X, Y)

                if plotting:
                    mean_error = calculate_abs_error(self.neural_network, X, Y)
                    plotter_q.put({
                        'mean_error': mean_error
                    })

                #print(f"Current error: {error}")
                
                if error < self.min_error:
                    self.min_error = error
                    self.min_weights_and_bias = self.neural_network.get_weights_and_bias()
                    #print('updated min_error', self.min_error)

                i += 1
                n += 1

            epoch_n += 1

        if plotting:
            plotter_q.put("STOP")
            print("Press 'q' to finish plot")
            keyboard.wait("q")

        return epoch_n >= epochs

    def get_best_model(self):
        return NeuralNetwork(self.min_weights_and_bias, self.hidden_layers, self.X_shape, self.Y_shape, self.activation_func, self.dx_activation_func)


class NeuralNetwork:

    def __init__(self, *args):

        if len(args) == 5:
            self.__init__1(args[0], args[1], args[2], args[3], args[4])
        else:
            self.__init__2(args[0], args[1], args[2], args[3], args[4], args[5])

    def __init__1(self, hidden_layers, X_shape, Y_shape, activation_func, dx_activation_func):
        # print(f'Hidden layers: {hidden_layers}')
        # print(X_shape)
        self.hidden_layers = hidden_layers
        self.X_shape = X_shape
        self.Y_shape = Y_shape
        self.activation_func = activation_func
        self.dx_activation_func = dx_activation_func
        self.network = self.create_network()

    def __init__2(self, weights_and_bias, hidden_layers, X_shape, Y_shape, activation_func, dx_activation_func):
        self.hidden_layers = hidden_layers
        self.X_shape = X_shape
        self.Y_shape = Y_shape
        self.activation_func = activation_func
        self.dx_activation_func = dx_activation_func
        self.network = self.create_network_with_weights_and_bias(weights_and_bias)

    def create_network_with_weights_and_bias(self, weights_and_bias):
        net = []
        for i in range(0, len(self.hidden_layers)):
            net.append([])
            for j in range(0, self.hidden_layers[i]):
                weights_len = self.X_shape
                if i > 0:
                    weights_len = self.hidden_layers[i-1]
                initial_weights = weights_and_bias[i][j]['weights']
                initial_bias = weights_and_bias[i][j]['bias']
                new_neuron = NetworkNeuron(initial_weights, initial_bias, self.activation_func)
                net[i].append(new_neuron)

        last_layer = []
        for k in range(0, self.Y_shape):
            weights_len = self.hidden_layers[-1]
            initial_weights = weights_and_bias[-1][k]['weights']
            initial_bias = weights_and_bias[-1][k]['bias']
            new_neuron = NetworkNeuron(initial_weights, initial_bias, self.activation_func)
            last_layer.append(new_neuron)

        net.append(last_layer)
        return net


    def create_network(self):

        net = []
        
        for i in range(0, len(self.hidden_layers)):
            net.append([])
            for j in range(0, self.hidden_layers[i]):
                weights_len = self.X_shape
                if i > 0:
                    weights_len = self.hidden_layers[i-1]
                initial_weights = np.random.uniform(-1, 1, weights_len)
                initial_bias = np.random.uniform(-1, 1)
                new_neuron = NetworkNeuron(initial_weights, initial_bias, self.activation_func)
                net[i].append(new_neuron)

        last_layer = []

        for k in range(0, self.Y_shape):
            weights_len = self.hidden_layers[-1] if len(self.hidden_layers) > 0 else self.X_shape
            initial_weights = np.random.uniform(-1, 1, weights_len)
            initial_bias = np.random.uniform(-1, 1)
            new_neuron = NetworkNeuron(initial_weights, initial_bias, self.activation_func)
            last_layer.append(new_neuron)

        net.append(last_layer)

        return net
            
    # backpropagation
    def apply_correction(self, learning_level, correction, entry):
        for l in range(0, len(self.network)):
            i = len(self.network)-1 - l
            if i == 0:
                entries = entry
            else:
                entries = np.array([n.last_activation for n in self.network[i-1]])

            for j in range(0, len(self.network[i])):
                neuron = self.network[i][j]
                if i == len(self.network) - 1:
                    # delta = correction[j] * self.dx_activation_func(neuron.last_excitation)
                    delta = correction[j]
                else:
                    result = 0
                    for k in range(0, len(self.network[i+1])):
                        parent_neuron = self.network[i+1][k]
                        result += parent_neuron.weights[j] * parent_neuron.last_delta
                    delta = self.dx_activation_func(neuron.last_excitation) * result
                
                self.network[i][j].apply_correction(learning_level, entries, delta)

        # last_correction_list = []
        # for j in range(0, len(entry)):
        #     last_correction = 0
        #     for k in range(0, len(self.network[0])):
        #         parent_neuron = self.network[0][k]
        #         last_correction += parent_neuron.weights[j] * parent_neuron.last_delta

        #     last_correction_list.append(last_correction)

        # return np.array(last_correction_list)
        
    def evaluate(self, entry):
        entries = entry
        for i in range(0, len(self.network)):
            if i > 0:
                entries = np.array([n.last_activation for n in self.network[i-1]])
            for j in range(0, len(self.network[i])):
                neuron = self.network[i][j]
                last = neuron.evaluate(entries)

                if i == len(self.network) - 1:
                    last = 1 if last >= 0 else -1
                    neuron.last_activation = last

        result = np.array([n.last_activation for n in self.network[-1]])

        if result.shape[0] == 1:
            result = result[0]
        
        return result

    def get_weights_and_bias(self):

        wab = []

        for i in range(0, len(self.network)):
            wab.append([])
            for j in range(0, len(self.network[i])):
                neuron = self.network[i][j]
                wab[i].append({
                    'weights': neuron.weights.copy(),
                    'bias': neuron.bias
                })

        return wab