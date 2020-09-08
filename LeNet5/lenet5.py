'''
LeNet5 by Pytorch
pooling:avg /f:tanh
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import numpy as np


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


def load_data(train_batch_size, test_batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose(
                           [transforms.Resize((32, 32)), transforms.ToTensor(),
                            transforms.Normalize((0.1307, ), (0.3081, ))]
                           )), batch_size=train_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, 
                       transform=transforms.Compose(
                           [transforms.Resize((32, 32)), transforms.ToTensor(), 
                            transforms.Normalize((0.1307, ), (0.3081,))]
                           )), batch_size=test_batch_size, shuffle=True)

    return (train_loader, test_loader)


def train(model, optimizer, epoch, train_loader):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train set, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def test(model, epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += loss_fn(output, target).data.item()
        pred = np.argmax(output.data, axis=1)
        correct = correct + np.equal(pred, target.data).sum()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set, Epoch {} , Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


model = LeNet5()

lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr)
train_batch_size = 256
test_batch_size = 1000
train_loader, test_loader = load_data(train_batch_size, test_batch_size)
epochs = 10

for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader)
    test(model, epoch, test_loader)
