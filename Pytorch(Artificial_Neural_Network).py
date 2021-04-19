import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        # Decide How Many Layers
        # InputLayer -> HiddenLayer(N) -> HiddenLayer(N) -> Output Layer(3 Classes)
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)   # Input Layer
        self.fc2 = nn.Linear(h1,h2)             # Hidden Layer
        self.out = nn.Linear(h2, out_features)  # Output Layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


torch.manual_seed(32)
model = Model()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Pytorch-Codes/iris.csv')
X = df.drop('target',axis=1).values
y = df['target'].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train) # Forward and get a prediction
    loss = criterion(y_pred, y_train)
    losses.append(loss) # To plot loss
    if i % 10 == 0:
        print('epoch {} and loss {}'.format(i, loss))

    # BackPropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(losses, 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss Plot')

# Evaluation on the Test Data

with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

print(loss)

# To get the Accuracy

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        print('{}.)  {}  {}  {}'.format(i+1,str(y_val),y_val.argmax().item(), y_test[i]))
        if y_val.argmax().item()== y_test[i]:
            correct += 1

print(f'The Number of Correct Examples on Test Data is {correct}')

# To save the Model
torch.save(model.state_dict(), 'Pytorch-Codes/Iris_Dataset_Pytorch.pt')

#To load the model
new_model = Model()
new_model.load_state_dict(torch.load('Pytorch-Codes/Iris_Dataset_Pytorch.pt'))
new_model.eval()


# To classify a single Flower

mystery_iris = torch.Tensor([5.6, 3.7, 2.2, 0.5])
with torch.no_grad():
    model.forward(mystery_iris)
    print(model.forward(mystery_iris).argmax())






