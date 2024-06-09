import numpy as np

# Read and preprocess the data
def read_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        temp_data = []
        for line in file:
            if line.strip() == "Falling" or line.strip() == "NotFalling":
                label = 0 if line.strip() == "NotFalling" else 1
                labels.append(label)
                data.append(temp_data)
                temp_data = []
            else:
                parts = line.strip().split(',')
                acc_values = list(map(float, parts))
                temp_data.extend(acc_values)
    data = np.array(data).reshape(-1, 1, 15)
    labels = np.array(labels)
    return data, labels

data, labels = read_data('trainingData.txt')
print(data[0])
