import torch
torch.__version__
import torch.nn as nn
import torch.nn.functional as F          # adds some efficiency
from torch.utils.data import DataLoader  # lets us load data in batches
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix  # for evaluating results
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_data = datasets.mnist.MNIST(root='', train=True, transform=transform, download=True)
test_data = datasets.mnist.MNIST(root= '',train=False, transform=transform, download=True)

train_data
test_data

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)


conv1 = nn.Conv2d(1,6,3,1)
conv2 = nn.Conv2d(6,16,3,1)

for i, (X_train, y_train) in enumerate(train_data):
    break

X_train.shape
x = X_train.view(1,1,28,28)
x = F.relu(conv1(x))
x.shape

x = F.max_pool2d(x,2,2)
x.shape
x = F.relu(conv2(x))
x.shape
x = F.max_pool2d(x,2,2)
x.shape
x.view(-1, 16*5*5).shape


class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1, 16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X,dim=1)

torch.manual_seed(42)
model = ConvolutionalNetwork()
model

for param in model.parameters():
    print(param.numel())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import time
start_time = time.time()
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    i+=1
    trn_crr = 0
    tst_crr = 0
    for b , (X_train, y_train) in enumerate(train_loader):
        b+=1
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_crr += batch_corr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if b % 600 == 0:
            print(f'Epoch: {i} Batch: {b} loss {loss.item()} ')
    train_losses.append(loss)
    train_correct.append(trn_crr)

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data,1)[1]
            tst_crr =+ (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_crr)

current_time = time.time()
total_time = current_time - start_time
print(f'The Training took {total_time} or {total_time/60} minutes')


plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title("Loss")
plt.legend()
plt.show()

plt.plot([t/600 for t in train_correct], label='training accuracy')
plt.plot([t/100 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()

test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    for X_test, y_test in test_load_all:
        correct = 0
        y_val = model(X_test)
        predicted = torch.max(y_val.data,1)[1]
        correct += (predicted == y_test).sum()

print(correct.item())
print(confusion_matrix(predicted.view(-1), y_test.view(-1)))