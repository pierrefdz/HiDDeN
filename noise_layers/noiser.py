

import re

import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from torchvision.transforms import functional

from noise_layers.jpeg_compression import JpegCompression
from noise_layers.jpeg_diff import DiffJPEG
from noise_layers.quantization import Quantization


def random_float(min, max):
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.

    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width


class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)

        noised_and_cover[0] = noised_image[
               :,
               :,
               h_start: h_end,
               w_start: w_end].clone()

        return noised_and_cover


class Cropout(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        super(Cropout, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        assert noised_image.shape == cover_image.shape

        cropout_mask = torch.zeros_like(noised_image)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        noised_and_cover[0] = noised_image * cropout_mask + cover_image * (1-cropout_mask)
        return  noised_and_cover


class Dropout(nn.Module):
    """
    Drops random pixels from the noised image and substitues them with the pixels from the cover image
    """
    def __init__(self, keep_ratio_range):
        super(Dropout, self).__init__()
        self.keep_min = keep_ratio_range[0]
        self.keep_max = keep_ratio_range[1]


    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        mask_percent = np.random.uniform(self.keep_min, self.keep_max)

        mask = np.random.choice([0.0, 1.0], noised_image.shape[2:], p=[1 - mask_percent, mask_percent])
        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
        # mask_tensor.unsqueeze_(0)
        # mask_tensor.unsqueeze_(0)
        mask_tensor = mask_tensor.expand_as(noised_image)
        noised_image = noised_image * mask_tensor + cover_image * (1-mask_tensor)
        return [noised_image, cover_image]


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


class Identity(nn.Module):
    """
    Identity-mapping noise layer. Does not change the image
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, noised_and_cover):
        return noised_and_cover


class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_range, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method


    def forward(self, noised_and_cover):

        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)

        return noised_and_cover


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


class Rotate(nn.Module):
    """
    Rotate the image by random angle between -degrees and degrees.
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


def parse_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)

def parse_crop(crop_command):
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))

def parse_resized_crop(resized_crop_command):
    matches = re.match(r'resized_crop\((\d+\.*\d*,\d+\.*\d*)\)', resized_crop_command)
    ratios = matches.groups()[0].split(',')
    scale_min = float(ratios[0])
    scale_max = float(ratios[1])
    return ResizedCrop((scale_min, scale_max))

def parse_cropout(cropout_command):
    matches = re.match(r'cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))

def parse_dropout(dropout_command):
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))

def parse_resize(resize_command):
    matches = re.match(r'resize\((\d+\.*\d*,\d+\.*\d*)\)', resize_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))

def parse_blur(blur_command):
    matches = re.match(r'blur\((\d*,\d*)\)', blur_command)
    ratios = matches.groups()[0].split(',')
    kernel_size_min = float(ratios[0])
    kernel_size_max = float(ratios[1])
    return Blur((kernel_size_min, kernel_size_max))

def parse_rotate(rotate_command):
    matches = re.match(r'rotate\((\d*)\)', rotate_command)
    degrees = float(matches.groups()[0])
    return Rotate(degrees)

def parse_jpeg(jpeg_command):
    matches = re.match(r'jpeg_diff\((\d*)\)', jpeg_command)
    quality = float(matches.groups()[0])
    return DiffJPEG(quality=quality)

def parse_attack_args(s):

    layers = [Identity()]
    split_commands = s.split('+')

    parse_dict = {
        'crop': parse_crop,
        'resized_crop': parse_resized_crop,
        'cropout': parse_cropout,
        'dropout': parse_dropout,
        'resize': parse_resize,
        'rotate': parse_rotate,
        'blur': parse_blur,
        'jpeg_diff': parse_jpeg,
        'jpeg': lambda x: 'JpegPlaceholder',
        'quant': lambda x: 'QuantizationPlaceholder',
    }

    for command in split_commands:
        command = command.replace(' ', '')
        command = command.replace('none', '')
        if len(command) != 0:
            attack = command.split('(')[0]
            if attack in parse_dict.keys():
                layers.append(parse_dict[attack](command))
            else:
                raise ValueError('Command not recognized: \n{}'.format(command))

    return layers


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        self.noise_layers = []
        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))
                elif layer == 'QuantizationPlaceholder':
                    self.noise_layers.append(Quantization(device))
                else:
                    raise NotImplementedError("%s does not exists"%layer)
            else:
                layer.to(device)
                self.noise_layers.append(layer)
        

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        return random_noise_layer(encoded_and_cover)

