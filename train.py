import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self, dicSize, outSize):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(dicSize, int(dicSize*1000))
        self.fc2 = nn.Linear(int(dicSize*1000), int(dicSize*200))
        self.fc3 = nn.Linear(int(dicSize*200), outSize)
   
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

    
#####################
####nn parameters####
#####################
batch_size = 2000
learning_rate = 0.0001
epochs = 10
log_interval = 5
#####################

fireDataset = torch.load("fireDataset")
fireDatasetSize = fireDataset[0][0].size(0)
fireOutClasses = len(set(fireDataset[-1][0].tolist()))
net = Net(fireDatasetSize, fireOutClasses+1).to(device)

# Осуществляем оптимизацию путем стохастического градиентного спуска
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
# Создаем функцию потерь
criterion = nn.NLLLoss()

train_loader = torch.utils.data.DataLoader(fireDataset, batch_size=batch_size, shuffle=True)
# запускаем главный тренировочный цикл
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data).type(torch.FloatTensor).to(device)
        target = Variable(target).type(torch.LongTensor).to(device)
        data = data.view(-1, fireDatasetSize)
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data))

torch.save(net, "net.model")


#####################################TEST#####################################################

fireTestDataset = torch.load("fireTestDataset")
fireTestDatasetSize = fireDataset[0][0].size(0)

test_loader = torch.utils.data.DataLoader(fireTestDataset, batch_size=batch_size, shuffle=True)
test_loss = 0
correct = 0
for data, target in test_loader:
    torch.no_grad()
    data = Variable(data).type(torch.FloatTensor).to(device)
    target = Variable(target).type(torch.LongTensor).to(device)
    data = data.view(-1, fireTestDatasetSize)
    net_out = net(data)
    # Суммируем потери со всех партий
    test_loss += criterion(net_out, target).data
    pred = net_out.data.max(1)[1]  # получаем индекс максимального значения
    correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))