import math
import numpy as np


# Define the activation functions
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    if x > 30:
        return 1
    if x < -30:
        return 0
    return 1 / (1 + np.exp(-x))


# Define the model architecture
class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


# Define the Dense layer
class Dense:
    def __init__(self, units, activation=None, weights=None, biases=None):
        self.units = units
        self.activation = activation
        self.weights = weights
        self.biases = biases

    def forward(self, x):
        z = x @ self.weights + self.biases  # Use matrix multiplication
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        else:
            return z


def main(name1, name2, name3):
    # Load the data from the file
    data = np.loadtxt(name1, delimiter=' ', dtype=str)
    a = []
    for d in data:
        a.append(float(d))
    data = np.array(a)
    weights = [data[:1024].reshape(16, 64),
                   data[1024:5120].reshape(64, 64),
                   data[5120:5184].reshape(64, 1),
                   data[5184:5248],
                   data[5248:5312],
                   data[5312:5313],
                   ]
    model = Sequential()
    biases1 = weights[3]
    model.add(Dense(64, activation='relu', weights=weights[0], biases=biases1))
    biases2 = weights[4]
    model.add(Dense(64, activation='relu', weights=weights[1], biases=biases2))
    biases3 = weights[5]
    model.add(Dense(1, activation='sigmoid', weights=weights[2], biases=biases3))
    result = []

    X = np.loadtxt(name2, delimiter=' ', dtype=str)
    X = np.array([list(map(int, x)) for x in X])
    for x in X:
        output = model.forward(x)
        if output > 0.5:
            result.append(1)
        else:
            result.append(0)
    with open(name3, 'w') as file:
        # Write each element of the list to a new line in the file
        for item in result:
            file.write(str(item) + '\n')


if __name__ == '__main__':
    main("wnet1.txt", "testnet1.txt", "result1.txt")

