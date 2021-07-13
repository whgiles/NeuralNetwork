import numpy as np


# logistical function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of the sigmoid (logistical) function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 + sigmoid(x))

# About: This is a dynamic Neural Network (NN) that can be customized at the user's discretion. The user can add hidden
# layers and decide on the amount of layer neurons. The input layer automatically conforms to the amount of features
# the data has. Each neuron activation is dependent on a sigmoid (logistic) function.
#
# Before Use: When init the NN, the training data must be in the matrix form, observations X features, where the
# last feature is the target. e.g. [[x1,x2,x3,... y],
#                                   [x1,x2,x3,... y]]
class NeuralNetwork:

    def __init__(self, train_data, learning_rate=.01, bias=0.0):
        self.learning_rate = learning_rate
        self.bias = bias
        self.train_data = train_data
        self.number_of_outputs = 1
        self.test_data = []
        self.layers = [len(self.train_data[0]), self.number_of_outputs]
        self.weights = ''

    # sets the number of output (hypothesis) variables in the last layer of the NN.
    # the default value is 1. If your network outputs are different, then this method must be used before you build
    # your model.
    def set_outputs(self, number_of_outputs):
        self.number_of_outputs = number_of_outputs

    # adds a new hidden layer in the NN to the left of the output layer. The "neurons" parameter is used to set the
    # number for nodes or features in the new layer.
    def add_layer(self, neurons):
        self.layers.insert(len(self.layers) - 1, neurons)

    # This is an optional method that takes a percentage of your training data, and uses it as test data to evaluate
    # your NN's accuracy. This is done randomly so bias does not enter the model.
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

    def randomize_weights(self):
