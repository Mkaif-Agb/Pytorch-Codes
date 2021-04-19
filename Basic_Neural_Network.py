import pandas as pd
import torch
df = pd.read_csv('Pytorch-Codes/iris.csv')
df.shape

'''
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()

plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa','Iris virginica','Iris versicolor']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
plt.show()

'''

X = df.iloc[:, :-1].values
y = df.iloc[:, -1:].values
# y = df['target'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

from torch.utils.data import TensorDataset, DataLoader
data = df.drop('target', axis=1).values
labels = df['target'].values
iris = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
type(iris)
# for i in iris:
#     print(i)

iris_loader = DataLoader(iris, batch_size=50, shuffle=True)
for batch in iris_loader:
    print(batch)


import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        # Decide How Many Layers
        # InputLayer -> HiddenLayer(N) -> HiddenLayer(N) -> Output Layer(3 Classes)
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
















