import torch
import torch.nn as nn

# This UNet is a bit different from the paper. This one is designed to allow for a few more parameters.
# These are:
# - The depth of the U can be altered
# - The number of channels as the convolutions are performed

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_features_start=64, u_depth=4):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = u_depth
        size = num_features_start

        self.conv = nn.Conv2d
        self.activation_func = nn.ReLU

        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        previous_size = self.in_channels
        current_size = size
        # down the U
        for _ in range(self.depth) :
            self.down_convs.append(nn.Sequential(
                self.conv(previous_size, current_size, kernel_size=3, stride=1, padding=1),
                self.activation_func(),
                self.conv(current_size, current_size, kernel_size=3, stride=1, padding=1),
                self.activation_func(),
            ))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2))
            previous_size = current_size
            current_size *= 2

        # bottom convolutions
        self.bottom = nn.Sequential(
            self.conv(previous_size, current_size, kernel_size=3, stride=1, padding=1),
            self.activation_func(),
            self.conv(current_size, current_size, kernel_size=3, stride=1, padding=1),
            self.activation_func()
        )

        # up the U
        for _ in range(self.depth) :
            next_size = current_size//2
            self.up_samples.append(nn.Sequential(
                nn.ConvTranspose2d(current_size, next_size, kernel_size=2, stride=2, padding=0),
                self.activation_func(),
            ))
            self.up_convs.append(nn.Sequential(
                self.conv(current_size, next_size, kernel_size=3, stride=1, padding=1),
                self.activation_func(),
                self.conv(next_size, next_size, kernel_size=3, stride=1, padding=1),
                self.activation_func(),
            ))
            current_size = next_size

        # last convolutional layer
        self.final = self.conv(current_size, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # go down the U
        activations = []
        for i in range(self.depth):
            x = self.down_convs[i](x)
            activations.append(x)
            x = self.down_samples[i](x)
        
        x = self.bottom(x)

        # back up the U
        for i in range(self.depth):
            x = self.up_samples[i](x)
            x = torch.cat((activations[-(i+1)], x), 1)
            x = self.up_convs[i](x)

        return self.final(x)


