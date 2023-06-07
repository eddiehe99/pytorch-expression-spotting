import torch
from torch import nn


class SOFTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # channel 1
        self.channel_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # channel 2
        self.channel_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=5, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # channel 3
        self.channel_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # interpretation
        self.interpretation = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.LazyLinear(out_features=400),
            nn.ReLU(),
            nn.LazyLinear(out_features=1),
        )

    def forward(self, inputs):
        inputs_1 = inputs[:, 0, :, :]
        inputs_1 = inputs_1.unsqueeze(1)
        inputs_2 = inputs[:, 1, :, :]
        inputs_2 = inputs_2.unsqueeze(1)
        inputs_3 = inputs[:, 2, :, :]
        inputs_3 = inputs_3.unsqueeze(1)
        # channel 1
        channel_1 = self.channel_1(inputs_1)
        # channel 2
        channel_2 = self.channel_2(inputs_2)
        # channel 3
        channel_3 = self.channel_3(inputs_3)
        # merge
        merged = torch.cat((channel_1, channel_2, channel_3), 1)
        # interpretation
        outputs = self.interpretation(merged)
        return outputs
