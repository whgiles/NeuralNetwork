import sys

import numpy as np
import matplotlib.pyplot as plt


# logistical function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of the sigmoid (logistical) function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 + sigmoid(x))


# About: This is a dynamic Neural Network (NN) for classification models that can be customized at the user's
# discretion. The user can add hidden layers and decide on the amount of layer neurons. The input layer automatically
# conforms to the amount of features the data has. Each neuron activation is dependent on a sigmoid (logistic) function.
#
# Before Use: When init the NN, the data must be in the matrix form, observations X features, where the
# last feature is the target. e.g. [[x1,x2,x3,... y],
#                                   [x1,x2,x3,... y]]
# (y) should be the index of the class from 0 to (n-1) where (n) equals the number of classes
class NeuralNetwork:

    def __init__(self, data, learning_rate=.01, bias=0.0):
        self.data = np.array(data)
        self.train_data = []
        self.target_data = []
        self.set_up_data()
        self.learning_rate = learning_rate
        self.bias = bias
        self.number_of_outputs = 0
        self.test_data = []
        self.layers = [len(self.train_data[0]), self.number_of_outputs]
        self.weights = []
        self.target_matrix = []
        self.cost = []

    # splits the data entry into target_data and train_data. Target data is the last column of data entry.
    def set_up_data(self):
        data = self.data.transpose()

        self.train_data = data[:-1].transpose()
        self.target_data = data[-1].reshape(len(self.data), 1)

        print("------------ data -----------------------")
        print("train data: ", np.shape(self.train_data))
        print(self.train_data)
        print("target data: ", np.shape(self.target_data))
        print(self.target_data)

    # sets the number of output (hypothesis) variables in the last layer of the NN.
    # This method must be called before you build your model.
    def set_outputs(self, number_of_outputs):
        self.number_of_outputs = number_of_outputs
        self.layers[len(self.layers) - 1] = number_of_outputs

        observations = len(self.train_data)
        I = np.identity(number_of_outputs)

        Y = np.zeros((observations, number_of_outputs))

        if number_of_outputs <= np.amax(self.target_data):
            print("ERROR: number of outputs is do small, or target data is incorrect. Remember target data entries "
                  "should be from 0 to (n-1)")
            sys.exit()

        for idx, val in enumerate(self.target_data):
            Y[idx] = I[val]

        self.target_matrix = Y

    # adds a new hidden layer in the NN to the left of the output layer. The "neurons" parameter is used to set the
    # number for nodes or features in the new layer.
    def add_layer(self, neurons):
        self.layers.insert(len(self.layers) - 1, neurons)

    # This is an optional method that takes a percentage of your training data, and uses it as test data to evaluate
    # your NN's accuracy. This is done randomly so bias does not enter the model.
    # if you call this method after randomize_weights(), then weights are reset and epsilon=1.
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

        self.randomize_weights()

    # randomizes weights for each layer of the NN, excluding the output layer.
    # Weights will be in the range of [-epsilon, epsilon]
    def randomize_weights(self, epsilon=1):
        weighted_layers = self.layers[:-1]

        weights = []
        print("------------ weights -----------------------")
        for idx, val in enumerate(weighted_layers):
            size = (self.layers[idx + 1], self.layers[idx])
            w = np.random.uniform(-1 * epsilon, epsilon, size)
            weights.insert(len(weights), w)
            print(size)
            print(w)
            print()
        self.weights = weights

    # This function calculates all (z) and (a) values of the NN for one iteration.
    # all_a[i] is the vector of activation values for the (i)th layer. all_a[i][j] is the activation value of the
    # (j)th neuron in the (i)th layer. This is analogous to all_z
    # a = sigmoid(z) for each (a) value in each layer.
    def forward_propagation(self):
        activation = self.train_data
        all_a = [activation]
        all_z = ['none']

        i = 0
        while i < len(self.layers) - 1:
            z = np.matmul(activation, np.transpose(self.weights[i])) + self.bias
            all_z.insert(len(all_z), z)

            activation = sigmoid(z)
            all_a.insert(len(all_a), activation)

            i += 1

        cost = self.cost_function(all_a[len(all_a) - 1])
        self.cost.insert(len(self.cost), cost)
        print("COST: ", cost)

        return all_a, all_z

    def back_propagation(self):
        all_a, all_z = self.forward_propagation()

        print("------------z values-----------------------")
        for idx, val in enumerate(all_z):
            print(np.shape(val))
            print("z", idx, ": ", val)
            print()

        print("------------a values-----------------------")
        for idx, val in enumerate(all_a):
            print(np.shape(val))
            print("a", idx, ": ", val)
            print()

        little_deltaL = all_a[len(all_a) - 1] - self.target_matrix
        weights = self.weights

        little_deltas = [little_deltaL]

        i = len(self.weights) - 1
        while i >= 1:
            little_delta = np.dot(little_deltas[0], weights[i]) * sigmoid_derivative(all_z[i])
            little_deltas.insert(0, little_delta)

            i -= 1

        print("------------little delta values-----------------------")
        for idx, val in enumerate(little_deltas):
            print(np.shape(val))
            print("little delta", idx, ": ", val)
            print()

        deltas = []
        print("number of a: ", len(all_a))
        print("number of little_deltas: ", len(little_deltas))
        j = 0
        while j < len(all_a) - 1:
            delta = np.dot(np.transpose(little_deltas[j]), all_a[j])
            deltas.insert(len(deltas), delta)
            j += 1

        print("number of Deltas: ", len(delta))
        print("------------DELTA values-----------------------")
        for idx, val in enumerate(deltas):
            print(np.shape(val))
            print("delta", idx, ": ", val)
            print()

        gradients = []
        i = 0
        for delta in deltas:
            gradient = (delta / len(self.train_data)) + (self.learning_rate / len(self.train_data)) * weights[i]
            gradients.insert(len(gradients), gradient)
            print("Gradients made: ", i + 1)

            i += 1

        print("------------ Gradients -----------------------")
        for idx, val in enumerate(gradients):
            print(np.shape(val))
            print("gradient", idx, ": ", val)
            print()

        print("=======================================================================================================")
        return gradients

    def train(self, iterations):

        j = 1
        while j <= iterations:
            gradients = self.back_propagation()
            i = 0
            for gradient in gradients:
                self.weights[i] = self.weights[i] - gradient
                i += 1

            j += 1

        print(self.weights)

    def cost_function(self, hypothesis):
        inner_function = -1 * self.target_matrix * np.log(hypothesis) - (
                (1 - self.target_matrix) * np.log(1 - hypothesis))
        return np.sum(inner_function)

    def plot(self):
        plt.plot(self.cost)
        plt.show()

    def reset(self):
        self.weights = []
        self.cost = []
