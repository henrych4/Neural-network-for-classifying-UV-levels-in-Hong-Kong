import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import torch.utils.data
from torch.autograd import Variable
import argparse

from net import Net

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--data', required=True)
args = parser.parse_args()

use_gpu = False
if torch.cuda.is_available() and args.cuda:
    use_gpu = True
    torch.cuda.set_device(0)

data = np.load(args.data)['data']
#data = np.random.permutation(data)

input_size = data.shape[1] - 1
hidden_size = 50
num_classes = 5
num_epochs = 100
batch_size = 10
learning_rate = 0.01
decay_factor = 0.5
max_train = 0
max_test = 0

train_set = data[:-1314]
test_set = data[-1314:]

train_set_x = train_set[:, :-1]
train_set_y = train_set[:, -1]
test_set_x = test_set[:, :-1]
test_set_y = test_set[:, -1]

tensor_train_x = torch.from_numpy(train_set_x).float()
tensor_train_y = torch.from_numpy(train_set_y).long()
tensor_test_x = torch.from_numpy(test_set_x).float()
tensor_test_y = torch.from_numpy(test_set_y).long()

train_dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=2,
            )
test_dataset = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)
test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=2,
            )

def validate(net, dataloader):
    correct = 0
    total = 0
    for data in dataloader:
        inputs, labels = data
        if use_gpu:
            outputs = net(Variable(inputs.cuda()))
            labels = labels.cuda()
        else:
            outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return correct/total*100

if use_gpu:
    net = Net(input_size, hidden_size, num_classes).cuda()
else:
    net = Net(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    if epoch % 10 == 0 and epoch > 0:
        learning_rate = learning_rate * decay_factor
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if (i+1) % 500 == 0:
            print('epoch: {}, iter: {}, loss: {}'.format(epoch+1 , i+1, running_loss/500))
            current_train = validate(net, train_dataloader)
            current_test = validate(net, test_dataloader)
            print('training accuracy: {}, test accuracy: {}'.format(current_train, current_test))
            if current_train > max_train:
                max_train = current_test
            if current_test > max_test:
                max_test = current_test
                best_model = net
            running_loss = 0

print('highest accuracy: {}'.format(max_test))
if best_model:
    print('saving best model...')
    torch.save(best_model, './model.pkl')
