
import os
import copy
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import style

from makegif import create_gif

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


train_dict = {}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Using GPU')
else:
    device = torch.device("cpu")
    print("Using CPU")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)


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
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train():
    for epoch in range(EPOCHS):
        train_dict[epoch] = {}
        net.train()
        for step, (images, labels) in enumerate(tqdm(train_loader)):
            train_dict[epoch][step] = {}
            X = images.to(device)
            y = labels.to(device)

            optimizer.zero_grad()
            outputs = net(X)
            loss = loss_function(outputs, y) 
            loss.backward()
            optimizer.step()

            for n, layer in enumerate(net.modules()):  
                if isinstance(layer, nn.Linear):
                    train_dict[epoch][step][n] = {}
                    train_dict[epoch][step][n]['weights'] = layer.weight.detach().cpu().clone()
                    train_dict[epoch][step][n]['biases'] = layer.bias.detach().cpu().clone()

        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')


train()


def test():
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():  
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy}%')

test()


with open("train_dict.pkl", "wb") as f:
    pickle.dump(train_dict, f)


for (images, label) in (test_loader):
    print(label[0])
    img = images[0]
    images = images.to(device)
    preds = net(images)
    print(f"PREDICTION: { torch.argmax(preds[0]) }")
    plt.imshow(img.reshape(28, 28))
    break

style.use("dark_background")

png_files = []

def visualize_layers(sample_num, fig):
    for n, (images, label) in enumerate(test_loader):
        output_truth = label[sample_num]
        input_data = images[sample_num]

        images = images.to(device)
        outputs = net(images)

        prediction = torch.argmax(outputs[sample_num])

        classes = { 0: "T-shirt/top", 
                    1: "Trouser", 
                    2: "Pullover", 
                    3: "Dress", 
                    4: "Coat", 
                    5: "Sandal", 
                    6: "Shirt", 
                    7:"Sneaker", 
                    8:"Bag", 
                    9:"Ankle boot"
        }

        if prediction == output_truth:
            title_text = f"[Truth: { classes[int(output_truth)] }]   [Prediction: { classes[int(prediction)] }]"
            fig.suptitle(title_text, fontsize=20, color='g')
        else: 
            title_text = f"[Truth: { classes[int(output_truth)] }]   [Prediction: { classes[int(prediction)] }]"
            fig.suptitle(title_text, fontsize=20, color='r')


        ax0 = plt.subplot2grid((1, 7), (0, 0), rowspan=1, colspan=1)
        ax0.set_xlabel('Input Image')
        ax1 = plt.subplot2grid((1, 7), (0, 1), rowspan=1, colspan=1)
        ax1.set_xticks([])
        ax1.set_xlabel('Layer#1')
        ax2 = plt.subplot2grid((1, 7), (0, 2), rowspan=1, colspan=1)
        ax2.set_xticks([])
        ax2.set_xlabel('ReLU')
        ax3 = plt.subplot2grid((1, 7), (0, 3), rowspan=1, colspan=1)
        ax3.set_xticks([])
        ax3.set_xlabel('Layer#2')
        ax4 = plt.subplot2grid((1, 7), (0, 4), rowspan=1, colspan=1)
        ax4.set_xticks([])
        ax4.set_xlabel('ReLU')
        ax5 = plt.subplot2grid((1, 7), (0, 5), rowspan=1, colspan=1)
        ax5.set_xlabel('Layer#3')
        ax6 = plt.subplot2grid((1, 7), (0, 6), rowspan=1, colspan=1)
        ax6.set_xlabel('Softmax')

        plt.yticks([i for i in range(10)], [classes[i] for i in range(10)])

        layers = []
        relu = nn.ReLU()

        for params in net.parameters():
            if len(params.shape) == 2:
                layer = params.sum(axis=1)
                relu_layer = layer

                relu_layer = [relu(relu_layer).detach().cpu().numpy()]
                layer = [layer.detach().cpu().numpy()]

                layers.append(layer)
                layers.append(relu_layer)

        ## remove the last relu layer because we will be using softmax
        layers = layers[:-1]
        ## Add the softmax layer
        layers.append([outputs[sample_num].detach().cpu().numpy()])

        layer_1 = np.rot90(layers[0], k=3, axes=(0, 1))
        layer_2 = np.rot90(layers[1], k=3, axes=(0, 1))
        layer_3 = np.rot90(layers[2], k=3, axes=(0, 1))
        layer_4 = np.rot90(layers[3], k=3, axes=(0, 1))
        layer_5 = np.rot90(layers[4], k=3, axes=(0, 1))
        layer_6 = np.rot90(layers[5], k=3, axes=(0, 1))


        ax0.imshow(input_data.reshape(28, 28))

        ax1.imshow(layer_1, cmap="RdYlGn")
        ax2.imshow(layer_2, cmap="RdYlGn")
        ax3.imshow(layer_3, cmap="RdYlGn")
        ax4.imshow(layer_4, cmap="RdYlGn")
        ax5.imshow(layer_5, cmap="RdYlGn")
        ax6.imshow(layer_6, cmap="RdYlGn")

        ax0.axis("off")

        # png_file = f"./images/{n}.png"
        # png_files.append(png_file)
        # plt.savefig(png_file)
        
    # create_gif(png_files, './images/layers.gif')

fig = plt.figure(figsize=(14, 7))
visualize_layers(0, fig)


import time

style.use("dark_background")

png_files = []

def make_plots():
    with open("train_dict.pkl", "rb") as f:
        train_dict = pickle.load(f)
    
    fig = plt.figure(figsize=(20, 8))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)

    epochs = 2

    for epoch in range(epochs):
        print("EPOCH", epoch)
        for step in train_dict[epoch]:
            if (step%25) == 0:
                # time.sleep(4)
                print("Step", step)

                weights_0 = train_dict[epoch][step][1]["weights"]
                weights_1 = train_dict[epoch][step][2]["weights"]
                weights_2 = train_dict[epoch][step][3]["weights"]


                ax0.imshow(np.rot90(weights_0, k=3, axes=(0, 1)), cmap="RdYlGn")
                ax1.imshow(weights_1, cmap="RdYlGn")
                ax2.imshow(np.rot90(weights_2, k=3, axes=(0, 1)), cmap="RdYlGn")

                fig.suptitle(f"Fashion MNIST Linear Layers. Epoch: {epoch+1} Step: {step}", fontsize=20)

                ax0.set_title("Layer 1 32x784")
                ax1.set_title("Layer 2 32x32")
                ax2.set_title("Layer 3 10x32")

                ax0.title.set_fontsize(10)
                ax1.title.set_fontsize(10)
                ax2.title.set_fontsize(10)

                # ax0.set_aspect(0.05)
                # ax0.set_xlim(0, 784)
                # ax0.set_ylim(0, 32)
                fig.set_size_inches(12, 12)


                # png_file = f"./images/{epoch}{step}.png"
                # png_files.append(png_file)
                # plt.savefig(png_file)


                plt.pause(0.01)
                ax0.clear()
                ax1.clear()
                ax2.clear()

    plt.show()
    # create_gif(png_files, './images/weights.gif')

make_plots()






