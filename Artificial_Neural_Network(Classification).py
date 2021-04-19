import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Pytorch-Codes/NYCTaxiFares.csv')
df['fare_amount'].describe()
df['fare_class'].value_counts()

def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers

    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = (r * c)  # in kilometers

    return d

df.columns
df['dist_km'] = haversine_distance(df, 'pickup_latitude','pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df.info()

my_time = df['pickup_datetime'][0]
df['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour'] < 12, 'am', 'pm')
df['Weekday'] = df['EDTdate'].dt.strftime('%a') # .dayofweek

cat_cols = ['Hour', 'AMorPM', 'Weekday']
df.columns
const_cols = ['pickup_longitude','pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','passenger_count', 'dist_km']
y_cols = ['fare_class']
df.dtypes

for cat in cat_cols:
    df[cat] = df[cat].astype('category')

df.dtypes
df['Hour'].head()
df['AMorPM'].head()
df['Weekday'].head()

# df['AMorPM'].cat.categories
# df['AMorPM'].cat.codes

hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values

cats = np.stack([hr, ampm, wkdy], axis=1)

# cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)

cats = torch.tensor(cats, dtype=torch.int64)
cats.shape

conts = np.stack([df[col].values for col in const_cols], axis=1)
conts = torch.tensor(conts, dtype=torch.float)
conts.shape

y = torch.tensor(df[y_cols].values, dtype=torch.float).reshape(-1, 1)
y.shape


cat_szs = [len(df[cols].cat.categories) for cols in cat_cols]
cat_szs
catz = cats[:2]
catz

emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
emb_szs

selfembeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
selfembeds

embeddingz = []
for i,e in enumerate(selfembeds):
    embeddingz.append(e(catz[:, i]))

embeddingz
z = torch.cat(embeddingz, 1)
z
selfembdrop = nn.Dropout(.4)
z = selfembdrop(z)
z



class TabularModel(nn.Module):
    def __init__(self,emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        layer_list = []
        n_embs = sum([nf for ni, nf in emb_szs])
        n_in = n_embs + n_cont
        for i in layers:
            layer_list.append(nn.Linear(n_in , i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(p))
            n_in = i

        layer_list.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))

        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x


torch.manual_seed(33)
model = TabularModel(emb_szs, conts.shape[1], 2, [200,100], p=0.4 )
model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 60000
test_size = int(0.2 * 60000)


cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size - test_size:batch_size]
const_train = conts[:batch_size - test_size]
const_test = conts[batch_size - test_size:batch_size]

cat_train.shape
cat_test.shape
const_train.shape

y_train = y[:batch_size - test_size]
y_test = y[batch_size-test_size:batch_size]
y_train.shape

import time
start = time.time()
epochs=300
losses = []
for i in range(epochs):
    i = i+1
    y_pred = model.forward(cat_train, const_train )
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    if i%10 == 1:
        print(f'Epoch {i} Loss {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
duration = time.time() - start
print(f'Training took {duration/60} Minutes')

plt.plot(losses)


with torch.no_grad():
    y_val = model(cat_test, const_test)
    loss = torch.sqrt(criterion(y_val, y_test))
print(loss)


for i in range(10):
    diff = np.abs(y_val[i].item() - y_test[i].item())
    print(f'{i} Predicted {y_val[i].item():8.2f} Original {y_test[i].item():8.2f} Difference {diff:8.2f}')


torch.save(model.state_dict(), 'Taximodel(Regression)ANN.pt')


