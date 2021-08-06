import json
import os
from pathlib import Path
import pprint
import argparse
import datetime
import time
import numpy as np
import torch
import pickle
import logging
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from hidden_configuration import *
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from noise_layers.noiser import parse_attack_args

import utils
import utils_train

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_parser():
    parser = argparse.ArgumentParser()

    # Data and checkpoint dirs
    parser.add_argument('--train-dir', default='/checkpoint/pfz/watermarking/data/train_coco_10k_resized', type=str)
    parser.add_argument('--val-dir', default='/checkpoint/pfz/watermarking/data/coco_1k_resized', type=str)
    parser.add_argument('--output_dir', default="", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=50, type=int)

    # Network params
    # parser.add_argument('--attack', nargs='*', action=NoiseArgParser, default="")
    parser.add_argument('--attacks', default="", type=str)
    # parser.add_argument('--attacks', default="crop((0.2,0.3),(0.4,0.5))+cropout((0.11,0.22),(0.33,0.44))+dropout(0.2,0.3)+jpeg()", type=str)
    parser.add_argument('--image_size', default=128, type=int)
    parser.add_argument('--num_bits', default=30, type=int)

    # Optimization params
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--enable_fp16', default=False, choices=[False], type=utils.bool_inst)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    # Distributed training parameters
    # parser.add_argument("--dist_url", default="env://", type=str)
    # parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--debug_slurm', action='store_true')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--master_port', default=-1, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')

    return parser


def train(args):
    # Distributed mode
    utils_train.init_distributed_mode(args)
    cudnn.benchmark = True

    # Set seeds for reproductibility 
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # Logger
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    print("__log__:%s" % json.dumps(vars(params)))
    
    # Git SHA
    print("git:{}".format(utils.get_sha()))

    # Preparing data
    train_loader, val_loader = utils_train.get_data_loaders(args.train_dir, args.val_dir, args.batch_size, args.num_workers)

    # Build HiDDeN and Noise model
    args.config_path = os.path.join(args.output_dir, 'config.json')
    if not os.path.exists(args.config_path):
        hidden_args = {'H':args.image_size, 'W':args.image_size,'message_length':args.num_bits,
            'encoder_blocks':4, 'encoder_channels':64,
            'decoder_blocks':7, 'decoder_channels':64,
            'use_discriminator':True,'use_vgg':False,
            'discriminator_blocks':3, 'discriminator_channels':64,
            'decoder_loss':1,'encoder_loss':0.7,
            'adversarial_loss':1e-3,'enable_fp16':args.enable_fp16                      
        } 
        attack_config = parse_attack_args(args.attacks)
        hidden_config = HiDDenConfiguration(**hidden_args)
        with open(args.config_path, 'w') as f:
            hidden_args['attacks_arg'] = args.attacks
            json.dump(hidden_args, f)
    else:
        with open(args.config_path, 'rb') as f:
            hidden_args = json.load(f)
            attack_config = parse_attack_args(hidden_args.pop('attacks_arg'))
            hidden_config = HiDDenConfiguration(**hidden_args)
    attacker = Noiser(attack_config, device)
    hidden_net = Hidden(hidden_config, device, attacker)

    # Distributed training
    hidden_net.encoder_decoder = nn.parallel.DistributedDataParallel(hidden_net.encoder_decoder, device_ids=[args.local_rank])
    hidden_net.discriminator = nn.parallel.DistributedDataParallel(hidden_net.discriminator, device_ids=[args.local_rank])

    # Optionally resume training
    args.checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    if os.path.exists(args.checkpoint_path):
        print('Loading checkpoint from file %s'%args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint['epoch']
        utils.model_from_checkpoint(hidden_net, checkpoint)
    else: 
        start_epoch = 0

    # Log
    if False:
        print('HiDDeN model: {}\n'.format(hidden_net.to_string()))
        print('Model Configuration:\n')
        print(pprint.pformat(vars(hidden_config)))
        print('\nNoise configuration:\n')
        print(pprint.pformat(str(attack_config)))

    # Train
    start_time = time.time()
    print("Starting HiDDeN training !")
    for epoch in range(start_epoch, args.epochs):
        print("========= Epoch %3i ========="%epoch)
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        # training one epoch
        train_stats = train_one_epoch(hidden_net, train_loader, epoch, args)
        val_stats = val_one_epoch(hidden_net, val_loader, epoch, args)

        save_dict = {
            'enc-dec-model': hidden_net.encoder_decoder.state_dict(),
            'enc-dec-optim': hidden_net.optimizer_enc_dec.state_dict(),
            'discrim-model': hidden_net.discriminator.state_dict(),
            'discrim-optim': hidden_net.optimizer_discrim.state_dict(),
            'epoch': epoch+1
        }

        utils_train.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils_train.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        log_stats_val = {**{f'val_{k}': v for k, v in val_stats.items()}, 'epoch': epoch}
        if utils_train.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                f.write(json.dumps(log_stats_val) + "\n")
                

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(hidden_net, data_loader, epoch, args):
    metric_logger = utils_train.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    for it, (imgs, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True) # BxCxHxW
        msgs = utils.generate_messages(imgs.size(0), args.num_bits).to(device).type(torch.float) # BxK
        losses, _ = hidden_net.train_on_batch([imgs, msgs])

        torch.cuda.synchronize()
        for name, loss in losses.items():
            metric_logger.update(**{name:loss})

    metric_logger.synchronize_between_processes()
    print(">>> Averaged train stats: ", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def val_one_epoch(hidden_net, data_loader, epoch, args):
    metric_logger = utils_train.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    for it, (imgs, _) in enumerate(metric_logger.log_every(data_loader, 50, header)):
        imgs = imgs.to(device, non_blocking=True) # BxCxHxW
        msgs = utils.generate_messages(imgs.size(0), args.num_bits).to(device).type(torch.float) # BxK
        losses, (encoded_images, attacked_images, decoded_messages) = hidden_net.validate_on_batch([imgs, msgs])
        torch.cuda.synchronize()
        for name, loss in losses.items():
            metric_logger.update(**{name:loss})

    metric_logger.synchronize_between_processes()
    print(">>> Averaged val stats: ", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    train(params)