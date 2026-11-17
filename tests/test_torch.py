import torch
import torch.nn as nn
import numpy as np

from tensorgrad import Tensor, Layer, MLP, unbroadcast, gradcheck


class Torch_MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 64)
        self.out = nn.Linear(64, 10)

    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.out(x)

        return x


def torch_grad(param):
    assert param.grad is not None
    return param.grad.numpy()


def test_torch():

    model = MLP()
    torch_model = Torch_MLP()
    criterion = nn.CrossEntropyLoss()
    test_x = np.random.rand(10, 784)
    test_y = np.eye(10)[np.random.randint(10, size=10)]

    torch_model.fc1.weight = torch.nn.Parameter(torch.tensor(model.fc1.weights.data).T)
    torch_model.fc2.weight = torch.nn.Parameter(torch.tensor(model.fc2.weights.data).T)
    torch_model.out.weight = torch.nn.Parameter(torch.tensor(model.out.weights.data).T)

    torch_model.fc1.bias = torch.nn.Parameter(torch.tensor(model.fc1.bias.data))
    torch_model.fc2.bias = torch.nn.Parameter(torch.tensor(model.fc2.bias.data))
    torch_model.out.bias = torch.nn.Parameter(torch.tensor(model.out.bias.data))

    model.zero_grad()
    torch_model.zero_grad()

    pred = model(Tensor(test_x))
    torch_pred = torch_model(torch.tensor(test_x))

    loss = pred.cross_entropy(test_y)
    torch_loss = criterion(torch_pred, torch.tensor(test_y))

    assert np.allclose(loss.data, torch_loss.detach().numpy(), rtol=1e-5, atol=1e-6)

    loss.backward()
    torch_loss.backward()

    assert np.allclose(model.fc1.weights.grad, torch_grad(torch_model.fc1.weight).T, rtol=1e-5, atol=1e-6)
    assert np.allclose(model.fc2.weights.grad, torch_grad(torch_model.fc2.weight).T, rtol=1e-5, atol=1e-6)
    assert np.allclose(model.out.weights.grad, torch_grad(torch_model.out.weight).T, rtol=1e-5, atol=1e-6)

    assert np.allclose(model.fc1.bias.grad, torch_grad(torch_model.fc1.bias), rtol=1e-5, atol=1e-6)
    assert np.allclose(model.fc2.bias.grad, torch_grad(torch_model.fc2.bias), rtol=1e-5, atol=1e-6)
    assert np.allclose(model.out.bias.grad, torch_grad(torch_model.out.bias), rtol=1e-5, atol=1e-6)
