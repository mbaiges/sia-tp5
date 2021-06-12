import numpy as np
import json
import math

from sklearn.datasets import make_classification

# np.random.seed(42)
# X, y = make_classification(n_samples=10, n_features=4, n_classes=2, n_clusters_per_class=1)
# y_true = y.reshape(-1, 1)

# print(X, y_true)
# print(X.shape, y_true.shape)

def get_data():
        
    f = open('data/fonts_training.json')
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
    X = X + 1
    
    return X, X

class DenseLayer:
    def __init__(self, n_units, input_size=None, activation='sigmoid', name=None):
        self.n_units = n_units
        self.input_size = input_size
        self.W = None
        self.name = name
        self.A = None  # here we will cache the activation values
        self.fn, self.df = self._select_activation_fn(activation)

    def __repr__(self):
        return f"Dense['{self.name}'] in:{self.input_size} + 1, out:{self.n_units}"

    def init_weights(self):
        self.W = np.random.randn(self.n_units, self.input_size + 1)

    @property
    def shape(self):
        return self.W.shape

    def _select_activation_fn(self, activation):
        if activation == 'relu':
            fn = lambda x: np.where(x < 0, 0.0, x)
            df = lambda x: np.where(x < 0, 0.0, 1.0)
        elif activation == 'sigmoid':
            fn = lambda x: 1 / (1 + np.exp(-x))
            df = lambda x: x * (1 - x)
        elif activation == 'tanh':
            fn = lambda x: (np.exp(x) - np.exp(-1)) / (np.exp(x) + np.exp(-x))
            df = lambda x: 1 - x**2
        elif activation == 'linear' or activation is None:
            fn = lambda x: x
            df = lambda x: 1.0
        else:
            NotImplementedError(f"Function {activation} cannot be used.")
        return fn, df

    def __call__(self, X):
        m_examples = X.shape[0]
        X_extended = np.hstack([np.ones((m_examples, 1)), X])
        Z = X_extended @ self.W.T
        A = self.fn(Z)
        self.A = A
        return A

    def backprop(self, delta, a):
        da = self.df(a)  # the derivative of the activation fn
        return (delta @ self.W)[:, 1:] * da

class SequentialModel:
    def __init__(self, layers, lr=0.01):
        self.lr = lr
        input_size = layers[0].n_units
        layers[0].init_weights()
        for layer in layers[1:]:
            layer.input_size = input_size
            input_size = layer.n_units
            layer.init_weights()
        self.layers = layers

    def __repr__(self):
        return f"SequentialModel n_layer: {len(self.layers)}"

    def forward(self, X):
        out = self.layers[0](X)
        for layer in self.layers[1:]:
            out = layer(out)
        return out

    @staticmethod
    def cost(y_pred, y_true):
        cost = y_pred - y_true
        # cost = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        return cost.mean()

    def _extend(self, vec):
        return np.hstack([np.ones((vec.shape[0], 1)), vec])

    def backward(self, X, y_pred, y_true):
        n_layers = len(self.layers)
        delta = y_pred - y_true
        a = y_pred

        dWs = {}
        for i in range(-1, -len(self.layers), -1):
            a = self.layers[i - 1].A

            dWs[i] = delta.T @ self._extend(a)
            delta = self.layers[i].backprop(delta, a)

        dWs[-n_layers] = delta.T @ self._extend(X)

        for k, dW in dWs.items():
            self.layers[k].W -= self.lr * dW


X, y_true = get_data() 

N = X.shape[1]
model = SequentialModel([
    DenseLayer(35, activation='relu', input_size=N, name='input'),
    DenseLayer(10, activation='relu', name='1st hidden'),
    DenseLayer(20, activation='relu', name='2nd hidden'),
    DenseLayer(35, activation='relu', name='output')
], lr=0.0001)

for e in range(100):
    for i in range(0, X.shape[0]):
        X_example = np.array([X[i]])
        y_example = np.array([y_true[i]])
        y_pred = model.forward(X_example)
        model.backward(X, y_pred, y_example)
        print(model.cost(y_pred, y_example))

print(X, y_true)
print(X.shape, y_true.shape)

predicts = model.forward(X)

for i in range(0, X.shape[0]):
    print('-----------------------------------------------------------------')
    print('Input')
    print(X[i])
    print('Output')
    print(predicts[i])
    print('Expected')
    print(y_true[i])
    print('-----------------------------------------------------------------')