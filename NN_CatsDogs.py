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

# OpenCV-Python is a library of Python bindings designed to solve computer vision problems.
import cv2
import numpy as np
# smart output processing bars
from tqdm import tqdm

# set to true if you haven't built your data or you want to rebuild you data
REBUILD_DATA = False


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


net = Net()
optimizer = optim.Adam(net.parameters(), lr=.001)
loss_function = nn.MSELoss()

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

BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i + BATCH_SIZE]

        net.zero_grad()
        output = net(batch_X)
        loss = loss_function(output, batch_y)
        loss.backward()
        optimizer.step()

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]

        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1

print('Accuracy: ', round(correct / total, 3))
try:
    torch.save(net, "/Users/williamgiles/repos/NeuralNetwork/netty")
except Exception as e:
    print(e)

new_net = torch.load("/Users/williamgiles/repos/NeuralNetwork/netty")
net_out = new_net(test_X[1].view(-1, 1, 50, 50))[0]
if net_out[0] > net_out[1]:
    print('cat')
else:
    print('dog')

image = np.array(test_X[1].view(-1, 1, 50, 50)[0][0])
print(image)
plt.imshow(image, cmap="gray")
plt.show()
