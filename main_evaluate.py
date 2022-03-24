
import argparse
import json
import logging
import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
from augly.image import functional as aug_functional
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional
from tqdm import tqdm

import decode
import encode
import utils
import utils_img
from model import Hidden
from noise_layers.noiser import Noiser
from model import HiDDenConfiguration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()

def get_parser():
    parser = argparse.ArgumentParser()

    # Experience parameters
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--name_exp", type=str, default="")
    parser.add_argument("--save_images", type=utils.bool_inst, default=False)
    parser.add_argument("--save_first_images", type=utils.bool_inst, default=True)
    parser.add_argument("--debug_mode", type=int, default=1)
    parser.add_argument("--debug_step", type=int, default=1)

    # Datasets parameters
    parser.add_argument("--data_dir", type=str, default="/checkpoint/lowik/watermarking/data/yfcc100m/", help="Folder directory")
    # parser.add_argument("--data_dir", type=str, default="/checkpoint/pfz/watermarking/data/coco_1k_resized", help="Folder directory")
    # parser.add_argument("--data_dir", type=str, default="/checkpoint/pfz/watermarking/data/coco_1k_orig", help="Folder directory")

    # Bits parameters
    parser.add_argument("--num_bits", type=int, default=30)
    parser.add_argument("--redundancy", type=int, default=1)
    parser.add_argument("--ecc_method", type=str, default="none", help="Should be 'none' or 'ldpc,dv=*,dc=*,snr=*'")

    # Model parameters
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--config_path", type=str)
    # parser.add_argument("--model_path", type=str, default="/private/home/pfz/HiDDeN/experiments/combined-noise/checkpoints/combined-noise--epoch-400.pyt")

    # Optimization parameters
    parser.add_argument("--batch_size", type=int, default=64)

    return parser


def evaluate(imgs, msgs, decoder, preprocessing, params, ecc_params, verbose=1, save_samples=True):
    
    attacks_dict = {
        "none": lambda x : x,
        "rotation": functional.rotate,
        "grayscale": functional.rgb_to_grayscale,
        "contrast": functional.adjust_contrast,
        "brightness": functional.adjust_brightness,
        "hue": functional.adjust_hue,
        "hflip": functional.hflip,
        "vflip": functional.vflip,
        "blur": functional.gaussian_blur, # sigma = ksize*0.15 + 0.35  - ksize = (sigma-0.35)/0.15
        "jpeg": aug_functional.encoding_quality,
        "resize": utils_img.resize,
        "center_crop": utils_img.center_crop,
        "meme_format": aug_functional.meme_format,
        "overlay_emoji": aug_functional.overlay_emoji,
        "overlay_onto_screenshot": aug_functional.overlay_onto_screenshot,
    }

    attacks = [{'attack': 'none'}] \
        + [{'attack': 'meme_format'}] \
        + [{'attack': 'overlay_onto_screenshot'}] \
        + [{'attack': 'rotation', 'angle': jj} for jj in range(0,45,5)] \
        + [{'attack': 'center_crop', 'scale': 0.1*jj} for jj in range(1,11)] \
        + [{'attack': 'resize', 'scale': 0.1*jj} for jj in range(1,11)] \
        + [{'attack': 'blur', 'sigma': 0.2*jj, 'kernel_size':41} for jj in range(1,21)] \
        + [{'attack': 'jpeg', 'quality': 10*jj} for jj in range(1,11)] \
        + [{'attack': 'contrast', 'contrast_factor': 0.5*jj} for jj in range(1,5)] \
        + [{'attack': 'brightness', 'brightness_factor': 0.5*jj} for jj in range(1,5)] \
        + [{'attack': 'hue', 'hue_factor': -0.5 + 0.25*jj} for jj in range(0,5)] \

        
    def generate_attacks(img, attacks):
        attacked_imgs = []
        for attack in attacks:
            attack = attack.copy()
            attack_name = attack.pop('attack')
            attacked_imgs.append(attacks_dict[attack_name](img, **attack))
        return attacked_imgs

    logs = []
    for ii, img in enumerate(tqdm(imgs)):
        
        attacked_imgs = generate_attacks(img, attacks)
        if ii==0 and save_samples:
            imgs_path = os.path.join(params.output_dir, 'imgs')
            if not os.path.exists(imgs_path):
                os.makedirs(imgs_path)
            for jj in range(len(attacks)):
                attacked_imgs[jj].save(os.path.join(imgs_path,"%i_%s.png"%(ii, str(attacks[jj])) ))
        decoded_data = decode.decode(attacked_imgs, decoder, preprocessing, params, ecc_params)

        for jj in range(len(attacks)):
            attack = attacks[jj].copy()
            attack_name = attack.pop('attack')
            param_names = ['param%i'%kk for kk in range(len(attack.keys()))]
            attack_params = dict(zip(param_names,list(attack.values()))) # change params name before logging
            decoded_datum = decoded_data[jj]
            diff = (~torch.logical_xor(msgs[ii], decoded_datum['msg'])).tolist()
            log = {
                "keyword": "result_transforms",
                "img": ii,
                "attack": attack_name,
                **attack_params,
                "msg_orig": msgs[ii].tolist(),
                "msg_decoded": decoded_datum['msg'].tolist(),
                "bit_acc": np.sum(diff)/len(diff),
            }
            logs.append(log)
            if verbose>1:
                logger.info("__log__:%s" % json.dumps(log))

    df = pd.DataFrame(logs).drop(columns='keyword')
    if verbose>0:
        logger.info('\n%s'%df)
    return df

def evaluate_psnr(imgs_out, imgs_orig):
    psnrs = []
    for ii in range(len(imgs_out)):
        psnrs.append(utils_img.psnr(np.asarray(imgs_out[ii], dtype=int), np.asarray(imgs_orig[ii], dtype=int)))
    return pd.DataFrame(psnrs, columns=['psnr'])

def read_eval_on_attacks_df(df, verbose=1):
    """
    Reads the dataframe output by the previous function and returns 
    average scores for each transformation
    """
    df['param0'] = df['param0'].fillna(-1)
    df_mean = df.groupby(['attack','param0'], as_index=False).mean().drop(columns='img')
    df_agg = df.groupby(['attack','param0'], as_index=False).agg(['mean','min','max','std']).drop(columns='img')

    if verbose>0:
        logger.info('\n%s'%df_mean)
        logger.info('\n%s'%df_agg)
                    
    return df_agg

def main(params):
    # Set seeds for reproductibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Logger
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logger.info("__log__:%s" % json.dumps(vars(params)))
    
    # Git SHA
    logger.info("git:{}".format(utils.get_sha()))

    # Load marking network
    with open(params.config_path, 'rb') as f:
        hidden_args = json.load(f)
        hidden_args.pop('attacks_arg')
        hidden_config = HiDDenConfiguration(**hidden_args)
    params.num_bits = hidden_config.message_length
    hidden_net = Hidden(hidden_config, device, Noiser([], device))
    checkpoint = torch.load(params.model_path)
    utils.model_from_checkpoint(hidden_net, checkpoint)
    encoder, decoder = hidden_net.encoder_decoder.encoder.eval(), hidden_net.encoder_decoder.decoder.eval()

    # Load images
    logger.info('Loading images...')
    default_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(params.data_dir, transform=default_transform)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, num_workers=8)

    # Initialize ECC
    ecc_params = utils.parse_params(params.ecc_method)
    if ecc_params['name']=='ldpc':
        import pyldpc
        d_v, d_c = int(ecc_params['dv']), int(ecc_params['dc'])
        ecc_params['H'], ecc_params['G'] = pyldpc.make_ldpc(params.num_bits, d_v, d_c, seed=0, systematic=True)
        params.msg_bits = ecc_params['G'].shape[1]
        logger.info("Using LDPC: (%i, %i) code" % (params.num_bits, params.msg_bits))
    elif ecc_params['name']=='none':
        params.msg_bits = params.num_bits

    # Encode
    logger.info('Watermarking images...')
    msgs_orig, msgs = utils.generate_messages(len(dataloader.dataset), params.msg_bits, ecc_params)
    pt_imgs_out = encode.encode(dataloader, msgs, encoder, params)
    imgs_out = [transforms.ToPILImage()(utils_img.unnormalize_img(pt_img).squeeze(0).clamp(0,1)) for pt_img in pt_imgs_out] 
    imgs_orig = [Image.open(dataloader.dataset.imgs[ii][0]) for ii in range(len(dataloader.dataset))]

    # Save images
    imgs_path = os.path.join(params.output_dir, 'imgs')
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
    if params.save_images:
        for ii in range(len(imgs_out)):
            imgs_orig[ii].save(os.path.join(imgs_path,"%i_orig.png"%0))
            imgs_out[ii].save(os.path.join(imgs_path,"%i_encoded.png"%ii))
    elif params.save_first_images:
        imgs_orig[0].save(os.path.join(imgs_path,"%i_orig.png"%0))
        imgs_out[0].save(os.path.join(imgs_path,"%i_encoded.png"%0))

    # Evaluate
    logger.info('Evaluating...')
    eval_on_attacks_df = evaluate(imgs_orig, msgs_orig, decoder, default_transform, params, ecc_params, verbose=params.debug_mode)
    eval_on_attacks_df_agg = read_eval_on_attacks_df(eval_on_attacks_df, params.debug_mode)
    eval_on_attacks_df_path = os.path.join(params.output_dir,'eval_on_attacks_df.csv')
    eval_on_attacks_df.to_csv(eval_on_attacks_df_path, index=False)
    eval_on_attacks_df_agg_path = os.path.join(params.output_dir,'eval_on_attacks_df_agg.csv')
    eval_on_attacks_df_agg.to_csv(eval_on_attacks_df_agg_path, index=False)
    logger.info('Succesfully saved DataFrame of evaluation on attacks')
    psnr_df = evaluate_psnr(imgs_out, imgs_orig)
    psnr_df_path = os.path.join(params.output_dir,'psnr_df.csv')
    psnr_df.to_csv(psnr_df_path, index=False)
    logger.info('Succesfully saved DataFrame of PSNRs')


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
