import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Using GPU')
else:
    device = torch.device("cpu")
    print("Using CPU")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)

transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

def load_data(data_type):
    X, y = [], []
    dataset_path = "fashion_mnist_images/" + data_type
    classes = os.listdir(dataset_path)
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                image = Image.open(img_path).convert('L')
                X.append(transform(image))
                y.append(int(class_name))
    return TensorDataset(torch.stack(X), torch.tensor(y))


train_dataset = load_data(data_type='train')
test_dataset = load_data(data_type='test')


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


EPOCHS = 5
BATCH_SIZE = 32
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

def train():
    for epoch in range(EPOCHS):
        net.train()
        for (images, labels) in tqdm(train_loader):
            X = images.to(device)
            y = labels.to(device)

            optimizer.zero_grad()
            outputs = net(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')


train()


