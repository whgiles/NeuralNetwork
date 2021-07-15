import numpy as np


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

    def set_up_data(self):
        data = self.data.transpose()

        self.train_data = data[:-1].transpose()
        self.target_data = data[-1].reshape(len(self.data), 1)

    # sets the number of output (hypothesis) variables in the last layer of the NN.
    # This method must be called before you build your model
    def set_outputs(self, number_of_outputs):
        self.number_of_outputs = number_of_outputs
        self.layers[len(self.layers) - 1] = number_of_outputs

        observations = len(self.train_data)
        I = np.identity(number_of_outputs)

        Y = np.zeros((observations, number_of_outputs))

        for idx, val in enumerate(self.target_data):

            Y[idx] = I[val]

        self.target_data = self.target_matrix

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
            print(z)
            print()
            i += 1

        return all_a, all_z

    def back_propagation(self):
        all_a, all_z = self.forward_propagation()

        # little delta (error) of the output Layer
        little_deltaL = all_a[len(all_a) - 3] - self
