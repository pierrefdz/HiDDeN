
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode(dataloader, msgs, encoder, params):
    """ 
    Watermark with HiDDeN encoder
    Args:
        pass
    Returns:
        Encoded images
    """
    msgs = torch.split(msgs, dataloader.batch_size, dim=0)
    pt_imgs_out = []
    pbar = enumerate(tqdm(dataloader))
    for batch, (imgs, _) in pbar:
        imgs = imgs.to(device, non_blocking=True) # BxCxWxH
        batch_msgs = msgs[batch].to(device, non_blocking=True).type(torch.float) # BxK
        encoded_imgs = encoder(imgs, batch_msgs) # BxCxWxH
        for ii, x in enumerate(encoded_imgs):
            pt_imgs_out.append(x.squeeze(0).detach().cpu())
    return pt_imgs_out

