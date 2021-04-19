import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

from PIL import Image
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

with Image.open('Pytorch-Codes/CATS_DOGS/test/CAT/9374.jpg') as im:
    display(im)

path = 'Pytorch-Codes/CATS_DOGS'
# img_names = []
# for folder, subfolders, filenames in os.walk(path):
#     for img in filenames:
#         img_names.append(folder + '/' + img)
# len(img_names)
#
# img_size = []
# rejected = []
#
# for items in img_names:
#     try:
#         with Image.open(items) as img:
#             img_size.append(img)
#     except:
#         rejected.append(items)
#
# len(img_size)
# len(rejected)
#
#
# df = pd.DataFrame(img_size)
# df[0].describe()

dog = Image.open('Pytorch-Codes/CATS_DOGS/train/DOG/4.jpg')
display(dog)
plt.imshow(dog)

dog.size
dog.getpixel((0,0))

transform = transforms.Compose([
    transforms.Resize((250,250)),
    transforms.CenterCrop(250),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])
im = transform(dog)
type(im)


plt.imshow(im)

plt.imshow(np.transpose(im.numpy(), (1,2,0)))  # to convert from tensor to images dimension


'''
For pretrained Network we use these values for transform
'''

transform = transforms([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




'''
Start Building the Model using pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

from PIL import Image
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')



train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root = 'Pytorch-Codes/CATS_DOGS'

train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root,'test'), transform=test_transform)
torch.manual_seed(42)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

class_names = train_data.classes
class_names

len(train_data)
len(test_data)


for images,labels in train_loader:
    break

images.shape

# Print the labels
print('Label:', labels.numpy())
print('Class:', *np.array([class_names[i] for i in labels]))

im = make_grid(images, nrow=5)  # the default nrow is 8

# Inverse normalize the images
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)

# Print the images
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.conv3 = nn.Conv2d(16,32,3,1)
        self.fc1 = nn.Linear(26*26*32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1, 26*26*32)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)


(((((224-2)/2)-2)/2)-2)/2
torch.manual_seed(101)
CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)



CNNmodel
for p in CNNmodel.parameters():
    print(p.numel())


import time
start_time = time.time()
epochs = 5
max_trn_batch= 800 # batch 10 * 800 ==== 8000 Images
max_tst_batch= 300 # Similarily 3000 Images
train_loss = []
test_loss = []
train_acc = []
test_acc = []

for i in range(epochs):

    trn_corr = 0
    tst_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):

        # if b == max_trn_batch:
        #     break
        b+=1

        y_pred = CNNmodel(X_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data,1)[1]
        corr = (predicted == y_train).sum()
        trn_corr += corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b %200 == 0 :
            print(f'Batch {b} Epochs {i} Loss {loss.item()}  Accuracy {trn_corr} ')
    train_loss.append(loss)
    train_acc.append(trn_corr)

    # Test Set

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
        #     if b == max_tst_batch:
        #         break
            y_val = CNNmodel(X_test)
            predicted = torch.max(y_val.data,1)[1]
            corr = (predicted == y_test).sum()
            tst_corr += corr
    loss = criterion(y_val, y_test)
    test_loss.append(loss)
    test_acc.append(tst_corr)

current_time = time.time()
total_time = current_time - start_time
print(f"The total time it took to train is {total_time} seconds or {total_time/60} Minutes")


plt.plot(train_loss, label='Training Loss',color='r')
plt.plot(test_loss, label='Test Loss', color='b')
plt.legend()
plt.title('Loss')
plt.show()


torch.save(CNNmodel.state_dict(),'Pytorch-Codes/Cats_Dogs_Pytorch.pt' )

plt.plot([t/80 for t in train_acc], label='training accuracy')
plt.plot([t/30 for t in test_acc], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend();

100* test_acc[-1].item()/6251

'''
PRE TRAINED NETWORKS

'''

Alexnetmodel = models.alexnet(pretrained=True )

for param in Alexnetmodel.parameters():
    param.required_grad = False
torch.manual_seed(42)
Alexnetmodel.classifier = nn.Sequential(nn.Linear(9216,1024),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(1024,2),
                                        nn.LogSoftmax(dim=1))

Alexnetmodel
for param in Alexnetmodel.parameters():
    print(param.numel())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Alexnetmodel.classifier.parameters(), lr=0.001)

import time

start_time = time.time()

epochs = 1

max_trn_batch = 800
max_tst_batch = 300

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        if b == max_trn_batch:
            break
        b += 1

        # Apply the model
        y_pred = Alexnetmodel(X_train)
        loss = criterion(y_pred, y_train)

        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b % 200 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{10 * b:6}/8000]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            if b == max_tst_batch:
                break

            # Apply the model
            y_val = Alexnetmodel(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed
torch.save(Alexnetmodel.state_dict(), 'Pytorch-Codes/Cats_dogs_Pytorch_Alexnet.pt')
