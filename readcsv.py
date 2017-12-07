import csv
import numpy as np

data = []
input_x = []
input_y = []

with open('v2_preprocessed.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        s = ', '.join(row)
        data.append(s)
data.pop(0)

for row in data:
    row = row.split(', ')
    tmpx = []

    '''
    ['Amount of Cloud', 'Dew Point Temp', 'Grass Min Temp', 'Max Temp', 
    'Mean Temp', 'Min Temp', 'Pressure', 'Rainfall', 
    'Relative Humidity', 'WET-Bulb Temp', 'Wind Speed', 'Bright Sunshine', 
    'Maximum UV Exposure Level']
    '''
    for x in row[:-1]:
        tmpx.append(float(x))

    input_x.append(tmpx)

    if row[-1] == 'Low':
        input_y.append(0)
    elif row[-1] == 'Moderate':
        input_y.append(1)
    elif row[-1] == 'High':
        input_y.append(2)
    elif row[-1] == 'Very High':
        input_y.append(3)
    elif row[-1] == 'Extreme':
        input_y.append(4)
    else:
        print('ERROR')

column_mean = np.mean(input_x, axis=0)
column_var = np.var(input_x, axis=0)
input_x = (input_x - column_mean) / np.sqrt(column_var)
input_y = [input_y]
input_x = np.asarray(input_x)
input_y = np.asarray(input_y)

input_xy = np.concatenate((input_x, input_y.T), axis=1)
np.savez("data_norm", data=input_xy)

