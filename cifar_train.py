import torch
import torchvision
import torchvision.transforms as transforms

import  torch.optim as optim
from model import *
#每个通道都要归一化
trainsform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset=torchvision.datasets.CIFAR10(root='./',train=True,download=True,transform=trainsform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset=torchvision.datasets.CIFAR10(root='./',train=False,download=True,transform=trainsform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes=('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net=CifarModel()

optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
criterion=nn.CrossEntropyLoss()

def test():
    pass
#train the network
EPOCH=100
for epoch in range(EPOCH):
    running_loss=0
    for i,data in enumerate(trainloader):
        img,label=data
        output=net(img)
        loss=criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss
        if(i%2000==1999):
            print('[%d,%5d] loss: %.3f' % (epoch+1,i+1,running_loss/2000))
            running_loss=0
    if(epoch%5==4):
        test()
    pass


print('Finished Training')
