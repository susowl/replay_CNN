'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from operator import itemgetter



import os
import argparse

from models import *
from utils import progress_bar
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



class Dataset_replay_batch():
    def __init__(self,replay_length=32):
        self.replay_length = replay_length
        self.basicDataset = basicDataset
        self.replayDB = []

    def put_batch(self,item,loss):
#       DB initialization. Adding first N batches.
        if len(self.replayDB)<self.replay_length:
            self.replayDB.append((item,loss))
            return
#       Add current batch.
#       Check if current batch already in replay DB
        for i in self.replayDB:
            if i[0] == item:    #update loss value
                i[1] = loss
                return

#       Check is current batch bad
        if(loss>self.replayDB[-1][1]):
            self.replayDB[-1] = (item,loss)
            self.replayDB.sort(key=itemgetter(1))

    def get_batch(self):
        return random.choice(self.replayDB)

    def __len__(self):
        return len(self.replayDB)

class Dataset_replay_items():
    def __init__(self,replay_length=32,batch_size=64):
        self.replay_length = replay_length
        self.batch_size = batch_size
        self.basicDataset = basicDataset
        self.replayDB = []

    def put_batch(self,item,loss):
#       DB initialization. Adding first N batches.

        batch_size = item[0].shape[0]
        for i in range(batch_size):
             self.put_item(item[0][i],loss[i])

    def put_item(self,item,loss):
        if len(self.replayDB) < self.replay_length:
            self.replayDB.append((item, loss))
            return
        #       Add current batch.
        #       Check if current batch already in replay DB
        for i in self.replayDB:
            if i[0] == item:  # update loss value
                i[1] = loss
                return

        #       Check is current batch bad
        if (loss > self.replayDB[-1][1]):
            self.replayDB[-1] = (item, loss)
            self.replayDB.sort(key=itemgetter(1))

    def get_batch(self):
        batchdata = random.choice(self.replayDB)
        for i in range(self.batch_size-1):
           batchdata = torch.cat((batchdata,random.choice(self.replayDB)))
        return batchdata
#        return random.choice(self.replayDB)

    def __len__(self):
        return len(self.replayDB)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

trainset_replay = Dataset_replay_batch()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG16')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
replaymode = True

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        def_inputs,def_targets = inputs, targets
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        trainset_replay.put_batch((def_inputs, def_targets),loss.item())

        if replaymode:
            inputs, targets = trainset_replay.get_batch()[0]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss += criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
