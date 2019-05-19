import torch
import os
import numpy as  np
import torchvision
import torchvision.transforms as transforms

import  torch.optim as optim
from model import *
#每个通道都要归一化
trainsform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset=torchvision.datasets.CIFAR10(root='./',train=True,download=True,transform=trainsform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset=torchvision.datasets.CIFAR10(root='./',train=False,download=True,transform=trainsform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=0)

classes=('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net=CifarModel()

optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
criterion=nn.CrossEntropyLoss()

def test():
    results=[]
    labels=[]
    for i,data in enumerate(testloader):
        img,label=data
        labels.extend(label)
        output=net(img)

        _, predicted = torch.max(output, 1)
        # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
        #                               for j in range(4)))
        results.extend(predicted)
    # 计算精度
    results=np.array(results)
    labels=np.array(labels)
    acc=np.equal(results,labels).sum()/results.shape[0]
    TP=((labels+results)==2).sum()
    FP=((labels-results)==-1).sum()
    FN=((labels-results)==1).sum()
    TN=((labels+results)==0).sum()
    rec=TP/(FN+TP)
    precision=TP/(FP+TP)

    return acc,precision,rec

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
        # pass
    if((epoch+1)%5==0):
        print("testing.........")
        acc, precision, rec=test()
        print("epoch %d: accuracy: %3f---precision: %.3f---recall: %.3f"%(epoch+1,acc,precision,rec))
        torch.save(net.state_dict(),os.path.join("./pretrained","epoch-"+str(epoch+1)+".pkl"))

print('Finished Training')
