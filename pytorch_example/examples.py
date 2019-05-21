from pytorch_example.my_relu import *

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N,D_in,H,D_out=64,1000,100,10
device = torch.device('cpu')
#create input data
x=torch.randn(N,D_in,device=device)
y=torch.randn(N,D_out,device=device)


#initialize weights
w1=torch.randn(D_in,H,device=device,requires_grad=True)
w2=torch.randn(H,D_out,device=device,requires_grad=True)

lr=1e-6

for epoch in range(10):
    pred = x.mm(w1).clamp(min=0).mm(w2)#clamp起到了relu激活函数的作用
    loss=(y-pred).pow(2).sum()
    print(epoch,'-',loss)#拿到python数据

    loss.backward()

    # Update weights using gradient descent. For this step we just want to mutate
    # the values of w1 and w2 in-place; we don't want to build up a computational
    # graph for the update steps, so we use the torch.no_grad() context manager
    # to prevent PyTorch from building a computational graph for the updates
    #在权重更新的时候是不需要计算图的,使用with torch,no_grad()
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

        # Manually zero the gradients after running the backward pass
        w1.grad.zero_()
        w2.grad.zero_()