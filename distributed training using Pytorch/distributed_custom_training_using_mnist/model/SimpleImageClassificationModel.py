import torch.nn as nn


class ConvolutionalNNModel(nn.Module):

    def __init__(self, input_channel=1, output_channel=10, need_BN=False):
        super(ConvolutionalNNModel, self).__init__()
        self.input_channel = input_channel
        self.output = output_channel
        self.need_BatchN = need_BN
        self.cnn1 = nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(32 * 4 * 4, output_channel)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
