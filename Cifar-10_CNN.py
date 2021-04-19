import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

transform = transforms.ToTensor()

train_data = datasets.cifar.CIFAR10(root='Pytorch-Codes/', train=True, transform=transform, download=True)
test_data = datasets.cifar.CIFAR10(root='Pytorch-Codes/', train=False, transform=transform, download=True)

train_data
test_data

torch.manual_seed(101)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)


class_names = ['plane', '  car', ' bird', '  cat', ' deer', '  dog', ' frog', 'horse', ' ship', 'truck']

for images, labels in train_loader:
    break

labels

np.set_printoptions(formatter=dict(int=lambda x: f'{x:5}')) # to widen the printed array

# Grab the first batch of 10 images
for images,labels in train_loader:
    break

# Print the labels
print('Label:', labels.numpy())
print('Class: ', *np.array([class_names[i] for i in labels]))

# Print the images
im = make_grid(images, nrow=5)  # the default nrow is 8
plt.figure(figsize=(10,4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))


class ConvolutionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5,1) # [0] is the number of channels, [1] is the number of filters, [2] is the kernel 5x5
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(6*6*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1, 6*6*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)


torch.manual_seed(101)

model = ConvolutionNetwork()
model

for param in model.parameters():
    print(param.numel())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



import time
start_time = time.time()

train_loss = []
test_loss = []
train_correct = []
test_correct = []
epochs=10

for i in range(epochs):

    trn_corr = 0
    tst_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        ###
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%1000 == 0:
            print(f'Epochs: {i} Batch: {b} Loss: {loss} Correct: {batch_corr}')

    train_loss.append(loss)
    train_correct.append(trn_corr)


    with torch.no_grad():

        for b, (X_test, y_test) in enumerate(test_loader):

            y_val = model(X_test)
            predicted = torch.max(y_val.data,1)[1]
            corr = (predicted == y_test).sum()
            tst_corr += corr

    loss = criterion(y_val, y_test)
    test_loss.append(loss)
    test_correct.append(tst_corr)

stop_time = time.time()
total = stop_time - start_time
print(f'Duration of the Model Took {total} seconds or {total/60} Minutes')


torch.save(model.state_dict(), 'Pytorch-Codes/Cifar_Pytorch.pt' )

plt.plot(train_loss, label='training loss')
plt.plot(test_loss, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend()

plt.plot([t/500 for t in train_correct], label='training accuracy')
plt.plot([t/100 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()

test_loader_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():

    correct = 0
    for (X_test, y_test) in test_loader_all:

        y_predd = model(X_test)
        predicted = torch.max(y_predd.data, 1)[1]
        corr = (predicted == y_test).sum()
        correct += corr

arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
df_cm = pd.DataFrame(arr, class_names, class_names)
plt.figure(figsize = (9,6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
plt.xlabel("prediction")
plt.ylabel("label (ground truth)")
plt.show()






