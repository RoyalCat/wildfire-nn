import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
from hyperdash import monitor
from hyperdash import Experiment


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class Net(nn.Module):

    def __init__(self, in_features, out_features):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, int(in_features/2))
        self.fc4 = nn.Linear(int(in_features/2), out_features)
   
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc4(x)
        return F.log_softmax(x)


def test(net, criterion, batch_size, test_dataset):
    fireTestDatasetSize = fireTestDataset[0][0].size(0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
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

    acc = int(correct)/len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * acc))
    return acc

@monitor("WildFire InfTraining")
def trainToAcc(needed_accuracy, train_dataset, test_dataset, net_file=None, exp=None):

    learning_rate = exp.param("learning rate", 0.00001)
    batch_size = exp.param("batch size", 4000)
    epoch_limit = exp.param("epoch limit", 100000)
    log_interval = exp.param("log interval", 15)

    momentum = exp.param("momentum", 0.9)

    InputSize = exp.param("Input size", int(train_dataset[0][0].size(0)))
    OutClasses = exp.param("Out Classes", len(set(train_dataset.tensors[1].tolist())))
    net = Net(InputSize, OutClasses+1).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    if net_file == None:
        epoch = 0
        accuracy = 0
    else:
        checkpoint = torch.load(net_file)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = test(net, criterion, batch_size, test_dataset)
        exp.param("loaded accuracy", accuracy)
        exp.metric("accuracy", accuracy)
        print("model: " + net_file + "loaded!")

    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    

    while accuracy < needed_accuracy and epoch < epoch_limit:
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data).type(torch.FloatTensor).to(device)
            target = Variable(target).type(torch.LongTensor).to(device)
            data = data.view(-1, InputSize)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data))
                #if loss < 10:
                exp.metric("tLoss", float(loss.data))
        if epoch % 2 == 0:             
            accuracy = test(net, criterion, batch_size, test_dataset)
            exp.metric("accuracy", accuracy)
            exp.metric("epoch", epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, "net.checkpoint")
        epoch += 1


fireDataset = torch.load("fireDataset")
fireTestDataset = torch.load("fireTestDataset")
trainToAcc(0.5, fireDataset, fireTestDataset)




