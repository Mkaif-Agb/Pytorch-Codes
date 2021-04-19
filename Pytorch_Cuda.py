import torch
torch.cuda.is_available()
torch.cuda.get_device_properties(0)
torch.cuda.get_device_name(0)
torch.cuda.current_device()
torch.cuda.memory_allocated()
torch.cuda.memory_cached()

a = torch.FloatTensor((1.0, 2.0, 3.0 ))
a.device

a = torch.FloatTensor((1.0, 2.0, 3.0)).cuda()
a.device



import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.out(X)

        return X

torch.manual_seed(42)
model = Model()

next(model.parameters()).is_cuda

gpu_model = Model().cuda()
next(gpu_model.parameters()).is_cuda

df = pd.read_csv('Pytorch-Codes/iris.csv')
X = df.drop('target',axis=1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33)

X_train = torch.FloatTensor(X_train).cuda()
X_test = torch.FloatTensor(X_test).cuda()
y_train = torch.LongTensor(y_train).cuda()
y_test = torch.LongTensor(y_test).cuda()

train_loader = DataLoader(X_train, batch_size=60, shuffle=True, pin_memory=True)
test_loader = DataLoader(X_test, batch_size=60, shuffle=False, pin_memory=True)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gpu_model.parameters(), lr=0.001)

import time
start_time = time.time()
epochs = 100
losses = []
total_corr = []
for i in range(epochs):

    trn_corr = 0
    y_pred = gpu_model(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    predicted = torch.max(y_pred.data,1)[1]
    batch_corr = (predicted == y_train).sum()
    trn_corr += batch_corr

    if i%10 == 0:
        print(f'Epochs {i} Loss {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



end_time = time.time() - start_time
print(f'Total Time {end_time}  Seconds or {end_time/60}  Minutes')
