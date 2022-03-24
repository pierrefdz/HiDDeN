
import torch
import numpy as np
import torch.nn.functional as F

from torchvision.transforms import functional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# image_mean = torch.Tensor(NORMALIZE_IMAGENET.mean).view(-1, 1, 1).to(device)
# image_std = torch.Tensor(NORMALIZE_IMAGENET.std).view(-1, 1, 1).to(device)

image_mean = 0.5
image_std = 0.5

def rgb2yuv(x):
    '''convert batched rgb tensor to yuv'''
    out = x.clone()
    out[:,0,:,:] =  0.299    * x[:,0,:,:] + 0.587    * x[:,1,:,:] + 0.114   * x[:,2,:,:]
    out[:,1,:,:] = -0.168736 * x[:,0,:,:] - 0.331264 * x[:,1,:,:] + 0.5   * x[:,2,:,:]
    out[:,2,:,:] =  0.5      * x[:,0,:,:] - 0.418688 * x[:,1,:,:] - 0.081312 * x[:,2,:,:]
    return out

def yuv2rgb(x):
    '''convert batched yuv tensor to rgb'''
    out = x.clone()
    out[:,0,:,:] = x[:,0,:,:] + 1.402 * x[:,2,:,:]
    out[:,1,:,:] = x[:,0,:,:] - 0.344136 * x[:,1,:,:] - 0.714136 * x[:,2,:,:]
    out[:,2,:,:] = x[:,0,:,:] + 1.772 * x[:,1,:,:]
    return out

def yuv2ych(x):
    '''convert batched yuv tensor to ych'''
    out = x.clone()
    out[:,0,:,:] = x[:,0,:,:]
    out[:,1,:,:] = (x[:,1,:,:]**2 + x[:,2,:,:]**2)**0.5
    out[:,2,:,:] = torch.atan2(x[:,2,:,:], x[:,1,:,:])/np.pi/2.
    #output[:,2,:,:] += 1 * (output[:,2,:,:] < 0).type(torch.float)
    return out

def ych2yuv(x):
    '''convert batched ych tensor to yuv'''
    out = x.clone()
    h = np.pi*x[:,2,:,:]*2.
    out[:,0,:,:] = x[:,0,:,:]
    out[:,1,:,:] = x[:,1,:,:]*torch.cos(h)
    out[:,2,:,:] = x[:,1,:,:]*torch.sin(h)
    return out

def rgb2ych(x):
    '''convert batched rgb tensor to ych'''
    yuv = rgb2yuv(x)
    return yuv2ych(yuv)
    
def ych2rgb(x):
    '''convert batched ych tensor to rgb'''
    yuv = ych2yuv(x)
    return yuv2rgb(yuv)

def rgba2ycha(x):
    '''convert batched rgba tensor to ycha'''
    a = x[:,3:4,:,:]
    ych = rgb2ych(x[:,0:3,:,:])
    return torch.cat([ych, a], dim=1)

def ycha2rgba(x):
    '''convert batched ycha tensor to rgba'''
    a = x[:,3:4,:,:]
    rgb = ych2rgb(x[:,0:3,:,:])
    return torch.cat([rgb, a], dim=1)

def rgba2yuva(x):
    '''convert batched rgba tensor to yuva'''
    a = x[:,3:4,:,:]
    yuv = rgb2yuv(x[:,0:3,:,:])
    return torch.cat([yuv, a], dim=1)

def yuva2rgba(x):
    '''convert batched yuva tensor to rgba'''
    a = x[:,3:4,:,:]
    rgb = yuv2rgb(x[:,0:3,:,:])
    return torch.cat([rgb, a], dim=1)

def ycha2yuva(x):
    '''convert batched ycha tensor to yuva'''
    a = x[:,3:4,:,:]
    yuv = ych2yuv(x[:,0:3,:,:])
    return torch.cat([yuv, a], dim=1)

def yuva2ycha(x):
    '''convert batched yuva tensor to ycha'''
    a = x[:,3:4,:,:]
    ych = yuv2ych(x[:,0:3,:,:])
    return torch.cat([ych, a], dim=1)

colorspace_functions = {
    ("rgb", "yuv"): rgb2yuv,
    ("rgb", "ych"): rgb2ych,
    ("yuv", "ych"): yuv2ych,
    ("yuv", "rgb"): yuv2rgb,
    ("ych", "yuv"): ych2yuv,
    ("ych", "rgb"): ych2rgb,
    ("rgba", "yuva"): rgba2yuva,
    ("rgba", "ycha"): rgba2ycha,
    ("yuva", "ycha"): yuva2ycha,
    ("yuva", "rgba"): yuva2rgba,
    ("ycha", "yuva"): ycha2yuva,
    ("ycha", "rgba"): ycha2rgba,
}

def convert(x, input_colorspace, output_colorspace, clip=True):
    if clip and (output_colorspace == "rgb" or output_colorspace == "rgba"):
        return F.hardtanh(colorspace_functions[(input_colorspace.lower(), output_colorspace.lower())](x), 0, 1)
    else:
        return colorspace_functions[(input_colorspace.lower(), output_colorspace.lower())](x)

class Colorspace(object):
    '''Convert from one colorspace to another
    
    Available colorspaces:
        RGB, RGBA
        YUV, YUVA
        YCH, YCHA
        
    Parameters:
        input_colorspace: string
            colorspace of input image tensor
        output_colorspace: string
            colorspace of output image tensor
        clip: bool
            clip RGB and RGBA outputs within 0-1
    '''
    def __init__(self, input_colorspace, output_colorspace, clip=True):
        self.input_colorspace = input_colorspace
        self.output_colorspace = output_colorspace
        self.clip = clip
    
    def __call__(self, tensor):
        if self.input_colorspace == self.output_colorspace:
            return tensor
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
            return convert(tensor, self.input_colorspace, self.output_colorspace, self.clip).squeeze(0)            
        else:
            return convert(tensor, self.input_colorspace, self.output_colorspace, self.clip)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '(input_colorspace={0}'.format(self.input_colorspace)
        format_string += ', output_colorspace={0}'.format(self.output_colorspace)
        format_string += ')'
        return format_string

def normalize_img(x):
    return (x.to(device) - image_mean) / image_std

def unnormalize_img(x):
    return (x.to(device) * image_std) + image_mean

def round_pixel(x):
    x_pixel = 255 * unnormalize_img(x)
    y = torch.round(x_pixel).clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

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

def psnr(x,y):
    return 20*np.log10(255) - 10*np.log10(np.mean((x-y)**2))

