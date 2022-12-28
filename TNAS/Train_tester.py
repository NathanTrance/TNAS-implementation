import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import TNAS.CELL
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

num_epochs = 2
batch_size = 256

trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR-10', train=True,
                                        download=True, transform=transform)
train_indices = range(25000)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False,sampler=train_indices)
#tao sẽ thêm phần validation sau

val_indices = range(25000,50000)
valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False,sampler=val_indices)

testset = torchvision.datasets.CIFAR10(root='./data/CIFAR-10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_metric(Arch):
    net = TNAS.CELL.TinyNetwork(16,5,Arch,10)
    criterion  = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9,nesterov=True,weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 200,eta_min=0)

    for epoch in range(num_epochs):
        net.train()
        scheduler.step()
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        correct = 0
        total = 0
        with torch.no_grad():
            net.eval()
            for inputs, labels in valloader: #về sau ở đây t sẽ thay bằng validation
                outputs = net(inputs)
                loss = criterion(outputs,labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                #đéo ai qtam đến loss làm metric nên ta cũng không có care đến in loss ra ngoài
            acc = correct/total
            print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    return acc

#CHẮC CHẮN CÓ BUG VÀ T CŨNG CHƯA THÊM PHẦN CHẠY TRÊN GPU ĐÂU NHA MÀY

# with torch.no_grad():
#         net.eval()
#         for inputs, labels in testset: #về sau ở đây t sẽ thay bằng validation
#             outputs = net(inputs)
#             loss = criterion(outputs,labels)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             #đéo ai qtam đến loss làm metric nên ta cũng không có care đến in loss ra ngoài
#         print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')