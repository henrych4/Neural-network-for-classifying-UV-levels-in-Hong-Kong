import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

from net import Net

best_model = torch.load('./model.pkl')

data = np.load('./data2_norm.npz')['data']

x = data[:, :-1]
y = data[:, -1]

tensor_x = torch.from_numpy(x).float()
tensor_y = torch.from_numpy(y).long()

dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=10,
                num_workers=2,
)

predict_y = torch.LongTensor()

for i, data in enumerate(dataloader, 0):
    inputs, _ = data
    outputs = best_model(Variable(inputs))
    _, predicted = torch.max(outputs.data, 1)
    predict_y = torch.cat((predict_y, predicted), 0)

predict_y_np = predict_y.numpy()

np.savetxt('predict_y.txt', predict_y_np, fmt='%i')
