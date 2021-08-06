from io import BytesIO
from PIL import Image
import torch
import numpy as np
from torch.autograd.variable import Variable
import torch.nn.functional as F

import augly.image as imaugs

from torchvision import transforms
from torchvision.transforms import functional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# image_mean = torch.Tensor(NORMALIZE_IMAGENET.mean).view(-1, 1, 1).to(device)
# image_std = torch.Tensor(NORMALIZE_IMAGENET.std).view(-1, 1, 1).to(device)

image_mean = 0.5
image_std = 0.5

def normalize_img(x):
    return (x.to(device) - image_mean) / image_std

def unnormalize_img(x):
    return (x.to(device) * image_std) + image_mean

def round_pixel(x):
    x_pixel = 255 * unnormalize_img(x)
    y = torch.round(x_pixel).clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

def project_linf(x, y, radius):
    """ Clamp x-y so that Linf(x,y)<=radius """
    delta = x - y
    delta = 255 * (delta * image_std)
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / image_std
    return y + delta

def psnr_clip(x, y, target_psnr):
    """ Clip x-y so that PSNR(x,y)=target_psnr """
    delta = x - y
    delta = 255 * (delta * image_std)
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2))
    if psnr<target_psnr:
        delta = (torch.sqrt(10**((psnr-target_psnr)/10))) * delta 
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2))
    delta = (delta / 255.0) / image_std
    return y + delta

def ssim_heatmap(img1, img2, window_size):
    """ Compute the SSIM heatmap between 2 images """
    _1D_window = torch.Tensor(
        [np.exp(-(x - window_size//2)**2/float(2*1.5**2)) for x in range(window_size)]
        ).to(device, non_blocking=True)
    _1D_window = (_1D_window/_1D_window.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(3, 1, window_size, window_size).contiguous())

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = 3)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = 3)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = 3) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = 3) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = 3) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map

def ssim_attenuation(x, y):
    """ attenuate x-y using SSIM heatmap """
    delta = x - y
    ssim_map = ssim_heatmap(x, y, window_size=17) # 1xCxHxW
    ssim_map = torch.sum(ssim_map, dim=1, keepdim=True)
    ssim_map = torch.clamp_min(ssim_map,0)
    # min_v = torch.min(ssim_map)
    # range_v = torch.max(ssim_map) - min_v
    # if range_v < 1e-10:
    #     return y + delta
    # ssim_map = (ssim_map - min_v) / range_v
    delta = delta*ssim_map
    return y + delta

def get_pixel_img(x):
    return torch.round(torch.clamp(255 * unnormalize_img(x), 0, 255))

def batch_l2(x):
    assert len(x.size())==4
    return (torch.mean(torch.norm(x, dim=[1,2,3], p=2))).item() # BxCxWxH -> B -> 1

def batch_linf(x):
    assert len(x.size())==4
    return (torch.mean(torch.norm(x, dim=[1,2,3], p=np.inf))).item() # BxCxWxH -> B -> 1

def batch_psnr(x):
    """ Compute mean PSNR on the batch """
    assert len(x.size())==4
    return torch.mean(20*np.log10(255) - 10*torch.log10(torch.mean(x**2, dim=[1,2,3]))).item() # BxCxWxH -> 1

def jpeg_compression(image, quality_factor):
    """
    Args:
        image: PIL image
        q: quality factor
    """
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality_factor, optimice=True)
    return Image.open(buffer)

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.size][::-1]
    return functional.center_crop(x, new_edges_size)

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.size][::-1]
    return functional.resize(x, new_edges_size)

def jpeg_compression_pt(pt_image, quality_factor):
    """
    Args:
        pt_image: normalized image tensor
        q: quality factor
    """
    pt_image = unnormalize_img(pt_image)
    image = transforms.ToPILImage()(pt_image.squeeze(0))
    image = jpeg_compression(image, quality_factor)
    image = transforms.ToTensor()(image).unsqueeze(0)
    return normalize_img(image)


if __name__ == '__main__':
    img1 = torch.randn(1, 3, 256, 256).to(device)
    img2 = torch.randn(1, 3, 256, 256).to(device)
    print(ssim_heatmap(img1, img2, window_size=11).size())
    print(ssim_heatmap(img1, img2, window_size=11).max())
    print(ssim_heatmap(img1, img2, window_size=11).min())