import numpy as np

from tensorgrad.engine import Tensor


class Layer:

    def __init__(self, nin, nout):
        self.weights = Tensor(np.random.randn(nin, nout) / nin ** 0.5)
        self.bias = Tensor(np.random.randn(nout) / nin ** 0.5)

    def __call__(self, x):
        return x @ self.weights + self.bias


class MLP:

    def __init__(self):
        self.fc1 = Layer(784, 512)
        self.fc2 = Layer(512, 64)
        self.out = Layer(64, 10)

        self.layers = (self.fc1, self.fc2, self.out)

    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.out(x)

        return x

    def parameters(self):
        return [param for l in self.layers for param in (l.weights, l.bias)]

    def step(self, learning_rate):
        for param in self.parameters():
            param.data -= learning_rate * param.grad

    def zero_grad(self):
        for param in self.parameters():
                param.grad *= 0
