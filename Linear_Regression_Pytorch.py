import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn

X = torch.linspace(1, 50, 50).reshape(-1, 1)
torch.manual_seed(71)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float32)
e
y = 2*X + 1 + e
y.shape

plt.scatter(X.numpy(), y.numpy())

'''
from sklearn.linear_model import Lasso
regressor = Lasso()
regressor.fit(X,y)
plt.scatter(X.numpy(), y.numpy())
plt.plot(X.numpy(), regressor.predict(X.numpy()))
'''

torch.manual_seed(59)
model = nn.Linear(in_features=1, out_features=1)
print(model.weight)
print(model.bias)

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


torch.manual_seed(59)
model = Model(1,1)
print(model.linear.weight)
print(model.linear.bias)

for name, param in model.named_parameters():
    print(name ,'\t',param.item())

x = torch.tensor([2.0])
print(model.forward(x))


x1 = np.linspace(0.0, 50.0, 50)
w1 = 0.1059
b1 = 0.9637
y1 = x1 * w1 + b1
print(y1)

plt.scatter(X.numpy(), y.numpy())
plt.plot(x1, y1, 'r')

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
epochs = 50
losses = []
for i in range(epochs):
    i += 1
    # Prediciting Forward Pass
    y_pred = model.forward(X)
    # Calculating Loss
    loss = criterion(y_pred, y)
    # Record Error
    losses.append(loss)
    print('Epoch {} Loss {} Weight {} Bias {}'.format(i, loss.item(),model.linear.weight.item(), model.linear.bias.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("MSE LOSS")

x = np.linspace(0.0, 50.0, 50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()
predicted_y = current_weight * x + current_bias
predicted_y
plt.scatter(X.numpy(), y.numpy())
plt.plot(x, predicted_y, 'r')






