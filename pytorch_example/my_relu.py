import torch
#继承自orch.autograd.Function类
class MyReLU(torch.autograd.function):
    @staticmethod
    def forward(cls,x):
        cls.sava_for_backward(x)
        return x.clamp(min=0)
    @classmethod
    def backward(cls,grad_output):
        """
            In the backward pass we receive the context object and a Tensor containing
            the gradient of the loss with respect to the output produced during the
            forward pass. We can retrieve cached data from the context object, and must
            compute and return the gradient of the loss with respect to the input to the
            forward function.
        """
        x=cls.saved_tensors()
        grad_x=grad_output.clone()
        grad_x[grad_x<0]=0
        return grad_x

