import torch
import torch.nn as nn
from hidden_configuration import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, config.encoder_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(config.encoder_channels, config.encoder_channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(config.encoder_channels + 3 + config.message_length, config.encoder_channels)

        self.final_layer = nn.Conv2d(config.encoder_channels, 3, kernel_size=1)

    def forward(self, image, message):

        expanded_message = message.unsqueeze(-1).unsqueeze(-1) # BxLx1x1
        expanded_message = expanded_message.expand(-1,-1, image.size(-2), image.size(-1)) # BxLxHxW

        encoded_image = self.conv_bns(image)
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w
