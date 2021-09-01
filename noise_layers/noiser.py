
from noise_layers.resized_crop import ResizedCrop
import re

import torch.nn as nn
import numpy as np
from torchvision import transforms

from noise_layers.rotate import Rotate
from noise_layers.blur import Blur
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.identity import Identity
from noise_layers.dropout import Dropout
from noise_layers.jpeg_diff import DiffJPEG
from noise_layers.resize import Resize
from noise_layers.quantization import Quantization
from noise_layers.jpeg_compression import JpegCompression

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

