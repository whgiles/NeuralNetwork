import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 + sigmoid(x))


class NeuralNetwork:

    def __init__(self, train_data, learning_rate=.01, bias=0.0):
        self.learning_rate = learning_rate
        self.bias = bias
        self.train_data = train_data
        self.number_of_outputs = 0
        self.test_data = []
        self.layers = [len(self.train_data[0]), self.number_of_outputs]

    def set_outputs(self, number_of_outputs):
        self.number_of_outputs = number_of_outputs

    def add_layer(self, neurons):
        self.layers.insert(len(self.layers) - 1, neurons)

    def set_test_data(self, percent_of_training_data=30):
        total_observations = len(self.train_data)
        test_observations = round(total_observations * (percent_of_training_data / 100))

        test_data = []
        i = 1
        while i <= test_observations:
            random_index = np.random.randint(0, len(self.train_data) - 1)
            observation = self.train_data[random_index]
            test_data.insert(len(test_data), observation)
            self.train_data = np.delete(self.train_data, random_index, axis=0)
            i += 1

        self.test_data = np.array(test_data)
