import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network
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

data, labels = read_data('./Epoch_size_study/trainingData_for_epoch.txt')

# Convert data to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Define the neural network, loss function, and optimizer
model = SmallCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the neural network
def train_model(model, criterion, optimizer, data, labels, epochs=400):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

train_model(model, criterion, optimizer, data_tensor, labels_tensor)

# Save weights and biases to a file
def save_weights_and_biases(model, file_path):
    with open(file_path, 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"{name}\n")
            if param.data.ndimension() == 2:  # 2D weights/biases
                np.savetxt(f, param.data.numpy())
            elif param.data.ndimension() == 1:  # 1D biases
                np.savetxt(f, param.data.unsqueeze(0).numpy())
            else:  # For 3D convolutional weights
                data = param.data.numpy()
                for i in range(data.shape[0]):
                    np.savetxt(f, data[i])

save_weights_and_biases(model, "./Epoch_size_study/trained_parameters_epoch_400.txt")