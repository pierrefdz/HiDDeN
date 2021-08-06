import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional

import numpy as np

class Blur(nn.Module):
    """
    Blur the image.
    """
    def __init__(self, kernel_size_scale):
        super(Blur, self).__init__()
        self.kernel_size_min, self.kernel_size_max = kernel_size_scale

    def forward(self, noised_and_cover):
        b = np.random.randint(self.kernel_size_min, self.kernel_size_max+1)
        b = b-(1-b%2)
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = functional.gaussian_blur(noised_image, b)
        
        return noised_and_cover
