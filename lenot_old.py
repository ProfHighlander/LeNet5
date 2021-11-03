import torch
from torch import nn
from torchvision import datasets, transforms


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

def train_module(model, learn_rate, momentum):
    loss = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    model.train()
    for epoch in range(10):
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
    print(loss_avg/size)


t = transforms.ToTensor()
mnist_trainset = datasets.MNIST(root='data', train=True, download=True, transform=t)
data = torch.utils.data.DataLoader(mnist_trainset, batch_size=50, drop_last=True)

model = LeNot5()

# Load progress
try:
    model = torch.load("./progress.pt")
    print("Loading a file")
except FileNotFoundError:
    print("Making a new file")


model.train()
for epoch in range(10):
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
    print(loss_avg/size)

# Saves progress in a file
torch.save(model, "./progress.pt")

mnist_testset = datasets.MNIST(root='data', train=False, download=True, transform=t)
data = torch.utils.data.DataLoader(mnist_testset, batch_size=50, drop_last=True)

model.eval()
right = 0
total = 0
for img, truth in data:
       

        #print(img[0])  
        output = model(img)

        right += sum(truth == torch.argmax(output, dim=1))
        total+=output.size(0)

print("Percent correct", (right/total))


print("End")