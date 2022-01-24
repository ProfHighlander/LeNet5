import torch
from torch import nn
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import argparse

class LeNot5(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.second_pool = nn.AvgPool2d(kernel_size=2)
        self.third_conv = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fourth_pool = nn.AvgPool2d(kernel_size=2)
        self.fifth_fully = nn.Linear(16*5*5, 120)
        self.sixth_fully = nn.Linear(120, 84)
        self.seventh_fully = nn.Linear(84, 10)
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = torch.relu(x)
        x = self.second_pool(x)
        x = self.third_conv(x)
        x = torch.relu(x)
        x = self.fourth_pool(x)
        #print(x.size())

        x = x.view(x.size(0), 16*5*5)

        x = self.fifth_fully(x)
        x = torch.relu(x)
        x = self.sixth_fully(x)
        x = torch.relu(x)
        x = self.seventh_fully(x)
        x = torch.relu(x)
        #x = nn.functional.log_softmax(x, dim=1)
        #print(x.size())
        return x

def visualize(model, layer=1):
    
    if layer == 1:
        kernels = model.first_conv.weight.detach().clone()
    else:
        kernels = model.third_conv.weight.detach().clone()

    nrow = 8 # Number of rows on chart
    padding = 1 # How much space divides the sqaures
    n,c,w,h = kernels.shape

    kernels = kernels.view(n*c, -1, w, h)
    rows = np.min((kernels.shape[0] // nrow + 1, 32))    
    grid = utils.make_grid(kernels, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow,rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.show()

def train_model(model, passes, learn_rate=0.09, momentum=0.9):
    print("Training")
    loss = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)


    model.train()
    for epoch in range(passes):
        loss_avg = 0
        size = 0
        for img, truth in data:
            #reset the gradients
            optim.zero_grad()
            #print(img.size())
            #print(truth.size())
            exp_truth = torch.zeros(truth.size(0), 10)
            #truth = torch.unsqueeze(truth, dim=1)
            #print(exp_truth.size())
            for idx in range(truth.size(0)):
                
                exp_truth[idx, truth[idx]] = 1
            #print(img[0])  
            output = model(img)
            #train the model
            #print(truth.size())
            #print(output)
            loss_num = loss(output, exp_truth)
            loss_num.backward()
            optim.step()
            loss_avg += loss_num.data
            size += 1
        print(str(epoch + 1) + ": " + str(loss_avg/size))
    return model

def test_model(model):
    mnist_testset = datasets.MNIST(root='data', train=False, download=True, transform=t)
    data = torch.utils.data.DataLoader(mnist_testset, batch_size=50, drop_last=True)
    
    model.eval()
    right = 0
    total = 0
    for img, truth in data:
        output = model(img)
        right += sum(truth == torch.argmax(output, dim=1))
        total+=output.size(0)

    print("Percent correct", (right/total))

def try_load(model, file):
    try:
        model = torch.load(file)
        print("Loaded", file)
        return True
    except FileNotFoundError:
        print(file, "does not exist, starting fresh")
        if input("Continue (y/n)? ") == "n":
            exit()
        return False


t = transforms.ToTensor()
mnist_trainset = datasets.MNIST(root='data', train=True, download=True, transform=t)
data = torch.utils.data.DataLoader(mnist_trainset, batch_size=50, drop_last=True)

# Argparse
parser = argparse.ArgumentParser(description="Options for the neural network.")
parser.add_argument("option", help="'train' to train, 'test' to test, 'both' for both")
parser.add_argument("-l", "--loadfile", help="Neural network to load")
parser.add_argument("-s", "--savefile", help="File to save to. Defaults to the file loaded")
parser.add_argument("-p", "--passes", help="How many times to run the train loop")
parser.add_argument("-v", "--visualize", help="'1' for first convolutional layer, '3' for second")
args = parser.parse_args()

model = LeNot5()
passes = 3 # How many times to run the train loop

if args.option == "train" or args.option == "both":
    if args.loadfile:
        if try_load(model, args.loadfile):
            model = torch.load(args.loadfile)
    if args.passes:
        passes = int(args.passes)
    model = train_model(model, passes)


if args.option == "test" or args.option == "both":
    if args.loadfile and args.option == "test":
        if try_load(model, args.loadfile):
            model = torch.load(args.loadfile)
        else:
            print("Could not find a file to test :(")
            exit()
    test_model(model)

if args.option == "train" or args.option == "both":
    if args.savefile:
        torch.save(model, args.savefile)
        print("Saved to", args.savefile)
    elif args.loadfile:
        if input("Save to " + args.loadfile + " (y/n)? ") == "y":
            torch.save(model, args.loadfile)
            print("Saved to", args.loadfile)

if args.visualize:
    #Git change for fun
    visualize(model, layer=int(args.visualize))

print("End")