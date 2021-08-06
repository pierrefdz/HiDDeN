import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional

import numpy as np

class Rotate(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, degrees, interpolation_method='nearest'):
        super(Rotate, self).__init__()
        self.degrees = degrees
        self.interpolation_method = interpolation_method

    def forward(self, noised_and_cover):
        rotation_angle = np.random.uniform(-self.degrees, self.degrees)
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = functional.rotate(noised_image, rotation_angle)

        return noised_and_cover
