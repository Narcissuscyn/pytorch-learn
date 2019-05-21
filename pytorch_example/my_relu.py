import torch
#继承自orch.autograd.Function类
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)#保存节点信息
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx,grad_output):
        """
            In the backward pass we receive the context object and a Tensor containing
            the gradient of the loss with respect to the output produced during the
            forward pass. We can retrieve cached data from the context object, and must
            compute and return the gradient of the loss with respect to the input to the
            forward function.
        """
        x,=ctx.saved_tensors
        grad_x=grad_output.clone()
        grad_x[x<0]=0#注意这里是x<0，而不是grad_output
        return grad_x

