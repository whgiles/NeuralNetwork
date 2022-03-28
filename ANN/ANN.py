import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

nn_architecture = [{'input_dim': 16, 'output_dim': 4, 'activation': 'sigmoid'},
                   {'input_dim': 4, 'output_dim': 5, 'activation': 'sigmoid'},
                   {'input_dim': 5, 'output_dim': 6, 'activation': 'sigmoid'},
                   {'input_dim': 6, 'output_dim': 1, 'activation': 'sigmoid'},
                   {'input_dim': 1, 'output_dim': 1, 'activation': 'sigmoid'}]


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def d_sigmoid(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


def cost_function(y_hat, y):
    m = y_hat.shape[1]
    cost = (-1 / m) * (np.dot(y, np.log(y_hat).T) + np.dot((1 - y), np.log(1 - y_hat).T))
    return np.squeeze(cost)


def init_layers(architecture, seed=99):
    np.random.seed(seed)
    params = {}

    for idx, layer in enumerate(architecture, start=1):
        index = idx
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']

        params['W' + str(index)] = np.random.randn(layer_output_size, layer_input_size)
        params['b' + str(index)] = np.ones((layer_output_size, 1))

    return params


def forward_propagation(X, params, architecture):
    A_curr = X
    memory = {}

    for idx, layer in enumerate(architecture, start=1):
        A_prev = A_curr
        index = idx
        weights = params['W' + str(index)]
        bias = params['b' + str(index)]
        Z_curr = np.dot(weights, A_prev) + bias
        A_curr = sigmoid(Z_curr)

        memory["A" + str(index)] = A_prev
        memory["Z" + str(index)] = Z_curr

    return A_curr, memory


def single_back_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev):
    m = A_prev.shape[1]

    dZ_curr = d_sigmoid(Z_curr) * dA_curr
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = 1 / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_back_propagation(Y_hat, Y, memory, params, architecture):
    grad_values = {}
    # m = Y.shape[1]

    dA_prev = -(np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for idx, layer in reversed(list(enumerate(architecture, start=1))):
        layer_index = idx

        dA_curr = dA_prev

        W_curr = params['W' + str(layer_index)]
        b_curr = params['b' + str(layer_index)]
        Z_curr = memory['Z' + str(layer_index)]
        A_prev = memory['A' + str(layer_index)]

        dA_prev, dW_curr, db_curr = single_back_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev)

        grad_values['dW' + str(layer_index)] = dW_curr
        grad_values['db' + str(layer_index)] = db_curr

    return grad_values


def update_params(params, gradient, architecture, learning_rate):
    for idx, layer in enumerate(architecture, start=1):
        params['W' + str(idx)] -= learning_rate * gradient['dW' + str(idx)]
        params['b' + str(idx)] -= + learning_rate * gradient['db' + str(idx)]

    return params


def train(X, Y, architecture, epochs, learning_rate):
    params = init_layers(architecture)
    cost_history = []
    for epoch in range(epochs):
        for idx, val in tqdm(enumerate(X), desc='EPOCH'):
            x = val.reshape(16,1)
            y = Y[idx]
            y_hat, cashe = forward_propagation(x, params, architecture)

            cost = cost_function(y_hat, y)
            cost_history.append(cost)

            grad_values = full_back_propagation(y_hat, y, cashe, params, architecture)
            params = update_params(params, grad_values, architecture, learning_rate)

    return params, cost_history


if __name__ == '__main__':
    df = pd.read_csv("training_data.csv").drop(columns='Unnamed: 0')
    df = df.drop(index=0).reset_index(drop=True)
    X = df.drop(columns=['Ethereum_close_t+1'])
    y = df['Ethereum_close_t+1'].reset_index(drop=True).to_frame()
    X, y = X.to_numpy(), y.to_numpy()
    parameters, costs = train(X, y, nn_architecture, 3, .001)

    plt.plot(costs)
    plt.title('MSE')
    plt.xlabel('iterations')
    plt.axvline(x=1000, color='r')
    plt.axvline(x=2000, color='r')
    plt.ylabel('cost')

    plt.show()
