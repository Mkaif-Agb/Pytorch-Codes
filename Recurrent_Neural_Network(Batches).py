import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = torch.linspace(0, 799, 800)
y = torch.sin(x*2*3.1416/40)
plt.plot(y.numpy())


test_size = 40
train_set = y[: -test_size]
test_set = y [-test_size:]
plt.plot(train_set, color='b', label='Training Set')
plt.plot(test_set, color='r', label= 'Test Set')
plt.legend()
plt.grid()
plt.tight_layout()


def input_data(seq, ws):
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:ws+i]
        label = seq[i+ws: i+ws+1]
        out.append((window, label))

    return  out

window_size = 40
train_data = input_data(train_set, window_size)
len(train_data)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = (torch.zeros(1,1, hidden_size), torch.zeros(1,1,hidden_size))

    def forward(self, seq):

        lstm_out , self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq),-1))

        return pred[-1]

torch.manual_seed(42)
model = LSTM()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

train_data[0]

for p in model.parameters():
    print(p.numel())


epochs = 10
future = 40

for i in range(epochs):

    for seq, y_train in train_data:
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1,1,model.hidden_size), torch.zeros(1,1,model.hidden_size))

        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    print(f'Epoch {i} Loss {loss.item()}')

    preds = train_set[-window_size:].tolist()
    for f in range(future):
        seq = torch.FloatTensor(preds[-window_size:])

        with torch.no_grad():
            model.hidden =  (torch.zeros(1,1,model.hidden_size), torch.zeros(1,1,model.hidden_size))
            preds.append(model(seq).item())
    loss = criterion(torch.tensor(preds[-window_size:]), y[760:])
    print(f'Loss {loss}')
    plt.plot(y.numpy())
    plt.plot(range(760,800), preds[window_size:])