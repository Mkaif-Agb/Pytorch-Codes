import torch
import torch.nn as nn
import torch.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

df = pd.read_csv('Pytorch-Codes/income.csv')
df['label'].value_counts()

df.columns
cat_cols = ['sex', 'education', 'marital-status',
       'workclass', 'occupation', 'income' ]
cont_cols = ['age', 'hours-per-week']
y_col = ['label']


len(cat_cols)
len(cont_cols)
len(y_col)


for cat in cat_cols:
    df[cat] = df[cat].astype('category')

df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)
df.head()


cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
emb_szs


sx = df['sex'].cat.codes.values
ed = df['education'].cat.codes.values
ms = df['marital-status'].cat.codes.values
wc = df['workclass'].cat.codes.values
oc = df['occupation'].cat.codes.values
ic = df['income'].cat.codes.values

cats = np.stack([sx, ed, ms, wc, oc, ic] ,axis=1)
cats
cats = torch.tensor(cats, dtype=torch.int64)

conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts

conts = torch.tensor(conts, dtype=torch.float)

y = torch.tensor(df[y_col].values).flatten()
y


b=30000
t=5000

cat_train = cats[:b-t]
cat_test = cats[b-t:b]
cont_train = conts[:b-t]
cont_test = conts[b-t:b]
y_train = y[:b-t]
y_test = y[b-t:b]


class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        # Call the parent __init__
        super().__init__()

        # Set up the embedding, dropout, and batch normalization layer attributes
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        # Assign a variable to hold a list of layers
        layerlist = []

        # Assign a variable to store the number of embedding and continuous layers
        n_emb = sum((nf for ni, nf in emb_szs))
        n_in = n_emb + n_cont

        # Iterate through the passed-in "layers" parameter (ie, [200,100]) to build a list of layers
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))

        # Convert the list of layers into an attribute
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        # Extract embedding values from the incoming categorical data
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        # Perform an initial dropout on the embeddings
        x = self.emb_drop(x)

        # Normalize the incoming continuous data
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)

        # Set up model layers
        x = self.layers(x)
        return x

torch.manual_seed(33)


model = TabularModel(emb_szs, conts.shape[1],2,[50], p=0.4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


import time
epochs=300
losses = []
start = time.time()
for i in range(epochs):
    i+=1
    y_pred = model.forward(cat_train, cont_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if i % 10 == 0:
        print(f'{i} Epochs {loss:2.4} Loss ')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
end = time.time()
print(f'{end - start / 60} Minutes to Complete the Training')

plt.plot(losses)


with torch.no_grad():
    y_val = model.forward(cat_test, cont_test)
    loss = criterion(y_val, y_test)
print(loss)

rows = len(y_test)
correct = 0
for i in range(rows):
    if y_val[i].argmax()== y_test[i]:
        correct += 1
print(100*correct/rows)
correct


