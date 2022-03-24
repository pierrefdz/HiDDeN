
import numpy as np
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode(imgs, decoder, preprocessing, params, ecc_params):
    """
    Decode images using HiDDeN decoder
    Args:
        imgs: Images to be decoded, list of PIL images 
        decoder: Decoder model
        preprocessing: Preprocessing applied before sending the image to the model
        params: params that contain angle of the hypercone
        ecc_params: Parameters of the ECC scheme

    Returns:
        List of decoded datum as a dictionary for each image
    """
    decoded_data = []
    for ii, img in enumerate(imgs):
        img = preprocessing(img).unsqueeze(0).to(device, non_blocking=True) # 1xCxHxW
        msg = 2*decoder(img)-1

        if ecc_params['name'] == "ldpc":
            import pyldpc
            msg = msg.cpu().numpy().astype(np.float64)
            if not 'snr' in ecc_params:
                ecc_params['snr'] = 10*np.log(1/np.abs(msg).var())
            msg = pyldpc.decode(ecc_params['H'], msg, snr=ecc_params['snr'], maxiter=100)  # K->K
            msg = pyldpc.get_message(ecc_params['G'], msg) # K->K'
            msg = torch.tensor(msg) > 0
        else:
            msg = torch.sign(msg).squeeze() > 0
        msg = msg.cpu()
        
        decoded_data.append({'index': ii, 'msg': msg})
    
    return decoded_data

