import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# For mac users to save plots
import matplotlib
matplotlib.use('Agg')

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

def train_model(model, criterion, optimizer, data, labels, epochs=13000):
    loss_values = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return loss_values

if __name__ == '__main__':
    data, labels = read_data('./Epoch_size_study/trainingData_for_epoch.txt')
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_values = train_model(model, criterion, optimizer, data_tensor, labels_tensor)

    plt.figure(figsize=(10, 6))
    plt.plot(range(13000), loss_values, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('./Epoch_size_study/loss_vs_epochs.png')
    plt.show()