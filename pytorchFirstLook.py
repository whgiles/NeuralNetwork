import torch
import torch.utils.data.dataloader
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim

train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# batch size is how many obs go through the model in one. batch size helps the accuracy model.
# low batch size take longer.
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# total = 0
# count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
#
# # data is a dic {a pixel by pixel matrix: the actual number drawn}
# for data in trainset:
#     Xs, ys = data
#     for y in ys:
#         count_dict[int(y)] += 1
#         total += 1
#
# print(count_dict)
# for i in count_dict:
#     print(f"{i}: {count_dict[i] / total * 100}")


class Net(nn.Module):

    def __init__(self):
        # init for nn.Module
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    # has to be called 'forward()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


net = Net()

# Adam is a loss reduction algorithm, similar to gradient decent, but generally better. It has a changing learning rate
optimizer = optim.Adam(net.parameters(), lr=0.001)

# EPOCHS is the general term for how many times the training data will do through the model
EPOCHS = 3
total_data = 0
for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        # makes gradients start at zero. idk why, guess its better
        net.zero_grad()
        output = net((X.view(-1, 28 * 28)))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()

# .train() and .eval() should be used somewhere. the tutorial didn't explain well


total = 0
correct = 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))