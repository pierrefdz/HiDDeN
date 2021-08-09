import torch.nn as nn
from hidden_configuration import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, config.discriminator_channels)]
        for _ in range(config.discriminator_blocks-1):
            layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_bns = nn.Sequential(*layers)
        self.linear = nn.Linear(config.discriminator_channels, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image):
        X = self.conv_bns(image)
        X = self.avg_pool(X) # BxDcx1x1
        X = X.squeeze(-1).squeeze(-1) # BxDc
        X = self.linear(X) # Bx2
        return self.softmax(X)[:,0:1] # Bx1 