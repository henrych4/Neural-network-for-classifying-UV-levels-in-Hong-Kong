import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_uniform(self.fc1.weight)
        nn.init.kaiming_uniform(self.fc2.weight)

    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.fc2(out)
        return out
