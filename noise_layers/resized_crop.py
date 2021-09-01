import torch
import torch.nn as nn
import numpy as np
import math

from torchvision.transforms import functional


class ResizedCrop(nn.Module):
    """
    Rotate the image by random angle between -degrees and degrees.
    """
    def __init__(self, crop_scale):
        super(ResizedCrop, self).__init__()
        self.crop_scale = crop_scale
        self.crop_ratio = (3/4, 4/3)

    def sample_params(self, x):
        width, height = functional._get_image_size(x)
        area = height * width
        log_ratio = torch.log(torch.tensor(self.crop_ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.crop_scale[0], self.crop_scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.crop_ratio):
            w = width
            h = int(round(w / min(self.crop_ratio)))
        elif in_ratio > max(self.crop_ratio):
            h = height
            w = int(round(h * max(self.crop_ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        i,j,h,w = self.sample_params(noised_image)
        noised_image = functional.crop(noised_image, i, j, h, w)
        noised_and_cover[0] = functional.resize(noised_image, noised_and_cover[1].shape[-2:])
        return noised_and_cover
