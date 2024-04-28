import torch
import torchvision
import torchvision.transforms as transforms



# 下载并加载CIFAR-10训练集
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)

# 下载并加载CIFAR-10测试集
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("ok")