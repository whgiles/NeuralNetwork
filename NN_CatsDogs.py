# this is a tutorial from:
# https://www.youtube.com/watch?v=9aYuQmMJvjA&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=6&t=721s
# This is a convolutional NN

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
# OpenCV-Python is a library of Python bindings designed to solve computer vision problems.
import cv2
import numpy as np
# smart output processing bars
from tqdm import tqdm

MODEL_NAME = f"model-{int(time.time())}"

# set to true if you haven't built your data or you want to rebuild you data
REBUILD_DATA = False

# if i decide to use a GPU later, I can.
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on the GPU')
    print('Available GPUs: ', torch.cuda.device_count())
else:
    device = torch.device('cpu')
    print('running on CPU')


class DogsVSCats:
    IMG_SIZE = 50
    CATS = '/Users/williamgiles/repos/NeuralNetwork/kagglecatsanddogs_3367a/PetImages/Cat'
    DOGS = '/Users/williamgiles/repos/NeuralNetwork/kagglecatsanddogs_3367a/PetImages/Dog'
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    # in the end, cat_count and dog_count should be about the same, otherwise the NN may be biased
    cat_count = 0
    dog_count = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            # os.listdir() makes a list of the items in that directory
            for f in tqdm(os.listdir(label)):
                try:
                    # concatenates label and f
                    path = os.path.join(label, f)
                    # turns image into gray-scale
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # converting to one-hot-vectors. np.eye() creates identity matrix, so when label = 0, eye is [1,0]
                    # and when label is 1, eye is [0,1]
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.DOGS:
                        self.dog_count += 1
                    elif label == self.CATS:
                        self.cat_count += 1

                # some of the images suck and, the method wont be able to run them
                except Exception as e:
                    pass

            # you must shuffle your obseravtions, to eliminate bias when training the NN
            np.random.shuffle(self.training_data)

            try:
                np.save('training_data.npy', self.training_data)
            except Exception as e:
                print(e)

            print('Dogs Count: ', self.dog_count)
            print('Cats Count: ', self.cat_count)


if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)


# print(len(training_data))
# print(training_data[2])
# plt.imshow(training_data[0][0], cmap="gray")
# plt.show()


# this part of the tutorial is in the link:
# https://www.youtube.com/watch?v=1gQR24B3ISE&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=6
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # making a 2 dimensional convolutional layer
        # the third parameter is the window size that the nn uses to examine the image
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # this a a random matrix that we will send through the conv layers to find the dimensions of the output
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        # for some reason it is hard to go from conv layers to linear layers, so the above operation is need
        # there is apparently a formula that is in the comment section of the tutorial
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=.001)
loss_function = nn.MSELoss()

print(MODEL_NAME)

# below uses a short hand for loop, which is equal to a list of training_data[0]
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X / 255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = .1
val_size = int(len(X) * VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]


def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True) / len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss


def test(size=32):
    X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = fwd_pass(X.view(-1, 1, 50, 50).to(device), y.to(device))
    return val_acc, val_loss


def train(net):
    BATCH_SIZE = 100
    EPOCHS = 5

    with open('model.log', 'a') as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50).to(device)
                batch_y = train_y[i:i + BATCH_SIZE].to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)
                if i % 50 == 0:
                    val_acc, val_loss = test(size=100)
                    f.write(
                        f"{MODEL_NAME},{round(time.time(), 3)},{round(float(acc), 2)},{round(float(loss), 4)},{round(float(val_acc), 2)},{round(float(val_loss), 4)}\n")


train(net)
