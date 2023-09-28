import torch
import torch.optim as optim
from SimpleModelClass import SimpleNetModel


def default_training_step():
    print("starts default_training_step on CPU")
    model = SimpleNetModel()

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    optimizer.zero_grad()
    input = torch.randn(20, 10)
    predictions = model(input)
    labels = torch.randn(20, 1)

    losses = loss_fn(labels, predictions)
    losses.backward()

    print("losses : ", losses)
    optimizer.step()
