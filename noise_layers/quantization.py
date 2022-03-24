import numpy as np
import torch
import torch.nn as nn


def transform(tensor, target_range):
    source_min = tensor.min()
    source_max = tensor.max()

    # normalize to [0, 1]
    tensor_target = (tensor - source_min)/(source_max - source_min)
    # move to target range
    tensor_target = tensor_target * (target_range[1] - target_range[0]) + target_range[0]
    return tensor_target


class Quantization(nn.Module):
    def __init__(self, device=None):
        super(Quantization, self).__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.min_value = 0.0
        self.max_value = 255.0
        self.N = 10
        self.weights = torch.tensor([((-1) ** (n + 1)) / (np.pi * (n + 1)) for n in range(self.N)]).to(device)
        self.scales = torch.tensor([2 * np.pi * (n + 1) for n in range(self.N)]).to(device)
        for _ in range(4):
            self.weights.unsqueeze_(-1)
            self.scales.unsqueeze_(-1)


    def fourier_rounding(self, tensor):
        shape = tensor.shape
        z = torch.mul(self.weights, torch.sin(torch.mul(tensor, self.scales)))
        z = torch.sum(z, dim=0)
        return tensor + z


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = transform(noised_image, (0, 255))
        # noised_image = noised_image.clamp(self.min_value, self.max_value).round()
        noised_image = self.fourier_rounding(noised_image.clamp(self.min_value, self.max_value))
        noised_image = transform(noised_image, (noised_and_cover[0].min(), noised_and_cover[0].max()))
        return [noised_image, noised_and_cover[1]]

