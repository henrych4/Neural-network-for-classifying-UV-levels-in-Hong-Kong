import numpy as np

from sklearn.metrics import classification_report

y = np.load('data2_norm.npz')['data'][:, -1]

f = open('predict_y.txt', 'r')
content = f.read().split('\n')
content.pop(-1)

predict_y = []
for row in content:
    predict_y.append(int(row))

predict_y = np.asarray(predict_y)

target_names = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']

print(classification_report(y, predict_y, target_names=target_names))

count_y = [0, 0, 0, 0, 0]
count_predict_y = [0, 0, 0, 0, 0]

for value in y:
    count_y[int(value)] = count_y[int(value)] + 1
for value in predict_y:
    count_predict_y[int(value)] = count_predict_y[int(value)] + 1

print(count_y)
print(count_predict_y)
