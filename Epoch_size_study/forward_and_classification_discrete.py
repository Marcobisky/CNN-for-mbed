import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the SmallCNN class
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(2 * 7, 4)
        self.fc2 = nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 2 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def load_parameters(model, file_path):
    state_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        key = lines[i].strip()
        i += 1
        values = []
        while i < len(lines) and not lines[i].strip().startswith('conv') and not lines[i].strip().startswith('fc'):
            values.append([float(x) for x in lines[i].strip().split()])
            i += 1
        values = np.array(values)
        state_dict[key] = torch.tensor(values, dtype=torch.float32).view(model.state_dict()[key].shape)
    
    model.load_state_dict(state_dict)

def generate_random_input():
    return np.random.uniform(-10, 10, 15)

def calculate_mean_and_variance(vector):
    mean = np.mean(vector)
    variance = np.var(vector)
    return mean, variance

def classify_and_plot(model, num_points=1000):
    plt.figure(figsize=(10, 6))
    
    for _ in range(num_points):
        input_vector = generate_random_input()
        mean, variance = calculate_mean_and_variance(input_vector)
        
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).view(1, 1, -1)
        output = model(input_tensor)
        falling_prob = output[0, 1].item()
        
        color = 'red' if falling_prob > 0.6 else 'blue'
        plt.scatter(mean, variance, color=color)
    
    plt.xlabel('Mean')
    plt.ylabel('Variance')
    plt.title('v-m plot')
    plt.show()

if __name__ == '__main__':
    model = SmallCNN()
    load_parameters(model, 'trained_parameters.txt')
    classify_and_plot(model)