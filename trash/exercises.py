import math
import numpy as np
import json
import os
import shutil
import ast
import re
import yaml

from models import AutoEncoder, MultilayerPerceptron
from utils import calculate_abs_error, calculate_mean_error
from activation_funcs import sign_activation, lineal_activation, tanh_activation, relu_activation, sigmoid_activation, dx_sign_activation, dx_lineal_activation, dx_tanh_activation, dx_relu_activation, dx_sigmoid_activation

from matplotlib import pyplot as plt

saves_folder_path = 'saves'

# Ej1

## data filenames
fonts_training = 'fonts_training.json'

config_filename = 'config.yaml'

with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    data_folder = config['data_folder']
    fonts_training = os.path.join(data_folder, config['fonts_training'])

class AutoEncoderExerciseTemplate:

    def __init__(self, Z_shape, hidden_layers1, activation_func1, dx_activation_func1, training_level1, hidden_layers2, activation_func2, dx_activation_func2, training_level2, epsilon, epochs, max_it_same_bias):
        self.Z_shape = Z_shape
        self.activation_func1 = activation_func1
        self.activation_func2 = activation_func2
        self.dx_activation_func1 = dx_activation_func1
        self.dx_activation_func2 = dx_activation_func2
        self.epsilon = epsilon
        self.epochs = epochs
        self.max_it_same_bias = max_it_same_bias
        self.training_level1 = training_level1
        self.training_level2 = training_level2
        
        self.hidden_layers1 = hidden_layers1 # las layers vienen en la forma [a, b, c, d...] donde cada letra representa las neuronas de cada capa
        self.hidden_layers2 = hidden_layers2

    def get_analysis_results(self, model, X, y):
        best_model = model.get_best_model()

        abs_error = calculate_abs_error(best_model, X, y)
        mean_error = calculate_mean_error(best_model, X, y)

        results = {
            'abs_error': float(abs_error),
            'mean_error': float(mean_error)
        }
        return results

    def build_results(self, perceptron, training_results, testing_results):
        best_model = perceptron.get_best_model()  

        if isinstance(best_model, list):
            weights_and_bias = []

            for model in best_model:
                model_weights_and_bias = model.get_weights_and_bias()

                if isinstance(model_weights_and_bias, list):
                    for i in range(0, len(model_weights_and_bias)):
                        for j in range(0, len(model_weights_and_bias[i])):
                            model_weights_and_bias[i][j]['weights'] = model_weights_and_bias[i][j]['weights'].tolist()
                            map(lambda e: float(e), model_weights_and_bias[i][j]['weights'])
                            model_weights_and_bias[i][j]['bias'] = float(model_weights_and_bias[i][j]['bias'])
                else:
                    model_weights_and_bias['weights'] = model_weights_and_bias['weights'].tolist()
                    map(lambda e: float(e), model_weights_and_bias['weights'])
                    model_weights_and_bias['bias'] = float(model_weights_and_bias['bias'])

                weights_and_bias.append(model_weights_and_bias)
        else:
            weights_and_bias = best_model.get_weights_and_bias()

            if isinstance(weights_and_bias, list):
                for i in range(0, len(weights_and_bias)):
                    for j in range(0, len(weights_and_bias[i])):
                        weights_and_bias[i][j]['weights'] = weights_and_bias[i][j]['weights'].tolist()
                        map(lambda e: float(e), weights_and_bias[i][j]['weights'])
                        weights_and_bias[i][j]['bias'] = float(weights_and_bias[i][j]['bias'])
            else:
                weights_and_bias['weights'] = weights_and_bias['weights'].tolist()
                map(lambda e: float(e), weights_and_bias['weights'])
                weights_and_bias['bias'] = float(weights_and_bias['bias'])

        results = {
            'configuration': weights_and_bias,
            'training': {
                'params': {
                    'epsilon': self.epsilon,
                    'epochs': self.epochs,
                    'max_it_same_bias': self.max_it_same_bias,
                    'training_level1': self.training_level1,
                    'training_level2': self.training_level2
                },
                'analysis': training_results
            }
        }
        if not testing_results is None:
            results['testing'] = { 'analysis': testing_results }

        return results

    # get new weights with a new training
    def train_and_test(self):
        X_train, Y_train, X_test, Y_test = self.get_data()

        # initialize 
        neural_net = AutoEncoder(X_train.shape[1], self.Z_shape, self.hidden_layers1, self.training_level1, self.activation_func1, self.dx_activation_func1, self.hidden_layers1, self.training_level1, self.activation_func1, self.dx_activation_func1)
       
        # train 
        print("Started training")
        result = neural_net.train(X_train, Y_train, self.epsilon, self.epochs, self.max_it_same_bias)

        if result:
            print(f"EPOCHS ({self.epochs}) passed")
            
        print("Finished training")

        training_results = self.get_analysis_results(neural_net, X_train, Y_train)

        # test 
        testing_results = None

        if X_test.shape[0] > 0:
            print("Started testing")
            testing_results = self.get_analysis_results(neural_net, X_test, Y_test)
            print("Finished testing")

        results = self.build_results(neural_net, training_results, testing_results)

        results_printing = json.dumps(results, sort_keys=False, indent=4, default=str)
        print(results_printing)

class Ej1(AutoEncoderExerciseTemplate):

    def __init__(self):
        hidden_layers1 = [3]
        hidden_layers2 = [3]
        Z_shape = 2
        epsilon = .001
        epochs = 200
        max_it_same_bias = 10000
        training_level1 = 0.01
        training_level2 = 0.01
        self.training_pctg = 1
        super().__init__(Z_shape, hidden_layers1, tanh_activation, dx_tanh_activation, training_level1, hidden_layers2, tanh_activation, dx_tanh_activation, training_level2, epsilon, epochs, max_it_same_bias)

    def get_data(self):
        
        f = open(fonts_training)
        data = json.loads(f.read())

        data1 = data['font1']

        def plain_data(matrix):
            arr = []
            for row in matrix:
                for el in row:
                    arr.append((el*2)-1)
            return arr

        X = []

        for el in data1:
            X.append(plain_data(el))

        X = np.array(X)

        print(X)

        # break_point = 1
        break_point = int(math.ceil(X.shape[0] * round(self.training_pctg, 1)))

        X_train = X[0:break_point]
        y_train = X[0:break_point]

        X_test = X[break_point:-1]
        y_test = X[break_point:-1]

        X_train

        return X_train, y_train, X_test, y_test

    def get_analysis_results(self, perceptron, X, y):
        best_model = perceptron.get_best_model()

        abs_error = calculate_abs_error(best_model, X, y)
        mean_error = calculate_mean_error(best_model, X, y)

        results = {
            'abs_error': float(abs_error),
            'mean_error': float(mean_error)
        }

        return results

    def build_results(self, perceptron, training_results, testing_results):
        results = super().build_results(perceptron, training_results, testing_results)

        results['training_pctg'] = self.training_pctg

        return results

class Exercise:

    def train_and_test(self):
        pass
    
class PerceptronExerciseTemplate(Exercise):

    def __init__(self, activation_func, epsilon, epochs, max_it_same_bias, training_level):
        self.activation_func = activation_func
        self.epsilon = epsilon
        self.epochs = epochs
        self.max_it_same_bias = max_it_same_bias
        self.training_level = training_level
 

    def get_data(self):
        pass

    def get_analysis_results(self, perceptron, X, y):
        best_model = perceptron.get_best_model()

        abs_error = calculate_abs_error(best_model, X, y)
        mean_error = calculate_mean_error(best_model, X, y)

        results = {
            'abs_error': float(abs_error),
            'mean_error': float(mean_error),

        }

        return results

    def build_results(self, perceptron, training_results, testing_results):
        best_model = perceptron.get_best_model()  

        weights_and_bias = best_model.get_weights_and_bias()

        if isinstance(weights_and_bias, list):
            for i in range(0, len(weights_and_bias)):
                for j in range(0, len(weights_and_bias[i])):
                    weights_and_bias[i][j]['weights'] = weights_and_bias[i][j]['weights'].tolist()
                    map(lambda e: float(e), weights_and_bias[i][j]['weights'])
                    weights_and_bias[i][j]['bias'] = float(weights_and_bias[i][j]['bias'])
        else:
            weights_and_bias['weights'] = weights_and_bias['weights'].tolist()
            map(lambda e: float(e), weights_and_bias['weights'])
            weights_and_bias['bias'] = float(weights_and_bias['bias'])

        results = {
            'configuration': weights_and_bias,
            'training': {
                'params': {
                    'epsilon': self.epsilon,
                    'epochs': self.epochs,
                    'max_it_same_bias': self.max_it_same_bias,
                    'training_level': self.training_level
                },
                'analysis': training_results
            }
        }

        if not testing_results is None:
            results['testing'] = { 'analysis': testing_results }

        return results

class MultilayerPerceptronExerciseTemplate(PerceptronExerciseTemplate):

    def __init__(self, hidden_layers, activation_func, dx_activation_func, epsilon, epochs, max_it_same_bias, training_level):
        super().__init__(activation_func, epsilon, epochs, max_it_same_bias, training_level)
        self.dx_activation_func = dx_activation_func
        self.hidden_layers = hidden_layers # las layers vienen en la forma [a, b, c, d...] donde cada letra representa las neuronas de cada capa

    # get new weights with a new training
    def train_and_test(self):
        X_train, Y_train, X_test, Y_test = self.get_data()

        print("--------------- X_train ---------------")
        print(X_train)
        print("--------------- Y_train ---------------")
        print(Y_train)
        print("--------------- X_test  ---------------")
        print(X_test)
        print("--------------- Y_test  ---------------")
        print(Y_test)
        
        # initialize perceptron
        Y_shape = 1 if len(Y_train.shape) == 1 else Y_train.shape[1]
        X_shape = X_train.shape[1]
        print(f'X_shape: {X_shape}, Y_shape: {Y_shape}')
        neural_net = MultilayerPerceptron(self.hidden_layers, X_shape, Y_shape, self.training_level, self.activation_func, self.dx_activation_func)
       
        # train perceptron
        print("Started training")
        result = neural_net.train(X_train, Y_train, self.epsilon, self.epochs, self.max_it_same_bias)

        if result:
            print(f"EPOCHS ({self.epochs}) passed")
            
        print("Finished training")
        
        training_results = self.get_analysis_results(neural_net, X_train, Y_train)

        # test perceptron
        testing_results = None

        if X_test.shape[0] > 0:
            print("Started testing")
            testing_results = self.get_analysis_results(neural_net, X_test, Y_test)
            print("Finished testing")

        results = self.build_results(neural_net, training_results, testing_results)

        results_printing = json.dumps(results, sort_keys=False, indent=4, default=str)
        print(results_printing)

        model = neural_net.get_best_model()

        original_images = []
        predicted_images = []

        for i in range(0, X_train.shape[0]):
            original_images.append(X_train[i].reshape((7,5)))
            # predicted_images.append(model.evaluate(X_train[i]).reshape((7,5)))

        for i in range(0, X_train.shape[0]):
            predicted_images.append(model.evaluate(X_train[i]).reshape((7,5)))
        
        # ext = original_images.extend(predicted_images)

        fig1, axes1 = plt.subplots(1, len(original_images), figsize=(7,5))
        for i, ax in enumerate(axes1.flat):
            ax.imshow(original_images[i])

        fig2, axes2 = plt.subplots(1, len(original_images), figsize=(7,5))
        for i, ax in enumerate(axes2.flat):
            ax.imshow(predicted_images[i])
        
        plt.show()
class Ej1v2(MultilayerPerceptronExerciseTemplate):

    def __init__(self):
        hidden_layers = [20, 10, 2, 10, 20]
        epsilon = .001
        epochs = 100
        max_it_same_bias = 500000
        training_level = 0.001
        self.training_pctg = 1
        super().__init__(hidden_layers, tanh_activation, dx_tanh_activation, epsilon, epochs, max_it_same_bias, training_level)

    def get_data(self):
        
        f = open(fonts_training)
        data = json.loads(f.read())

        data1 = data['font1']

        def plain_data(matrix):
            arr = []
            for row in matrix:
                for el in row:
                    arr.append(el)
            return arr

        X = []

        for el in data1:
            X.append(plain_data(el))

        X = np.array(X)
        X = 2*X - 1

        print(X)

        # break_point = 15
        break_point = int(math.ceil(X.shape[0] * round(self.training_pctg, 1)))

        X_train = X[0:break_point]
        y_train = X[0:break_point]

        X_test = X[break_point:-1]
        y_test = X[break_point:-1]

        X_train

        return X_train, y_train, X_test, y_test

    def get_analysis_results(self, perceptron, X, y):
        best_model = perceptron.get_best_model()

        abs_error = calculate_abs_error(best_model, X, y)
        mean_error = calculate_mean_error(best_model, X, y)

        results = {
            'abs_error': float(abs_error),
            'mean_error': float(mean_error)
        }

        return results

    def build_results(self, perceptron, training_results, testing_results):
        results = super().build_results(perceptron, training_results, testing_results)

        results['training_pctg'] = self.training_pctg

        return results