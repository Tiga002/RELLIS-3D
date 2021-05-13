"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import absolute_import
from __future__ import division
import argparse
from functools import partial
from config import cfg, assert_and_infer_cfg
import logging
import math
import os
import sys
import time

import torch
import numpy as np

from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
from utils.f_boundary import eval_mask_boundary
import datasets
import loss
import network
import optimizer
from tqdm import tqdm
from PIL import Image
from visualization import Colorize
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms

# Argument Parser
parser = argparse.ArgumentParser(description='GSCNN')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.gscnn.GSCNN')
parser.add_argument('--dataset', type=str, default='cityscapes')
parser.add_argument('--num_classes', type=int, default='19')
parser.add_argument('--cv', type=int, default=0,
                    help='cross validation split')
parser.add_argument('--joint_edgeseg_loss', action='store_true', default=True,
                    help='joint loss')
parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class')
parser.add_argument('--eval_thresholds', type=str, default='0.0005,0.001875,0.00375,0.005',
                    help='Thresholds for boundary evaluation')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Rescaled LR Rate')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Rescaled Poly')

parser.add_argument('--edge_weight', type=float, default=1.0,
                    help='Edge loss weight for joint loss')
parser.add_argument('--seg_weight', type=float, default=1.0,
                    help='Segmentation loss weight for joint loss')
parser.add_argument('--att_weight', type=float, default=1.0,
                    help='Attention loss weight for joint loss')
parser.add_argument('--dual_weight', type=float, default=1.0,
                    help='Dual loss weight for joint loss')

parser.add_argument('--evaluate', action='store_true', default=False)

parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--sgd_finetuned',action='store_true',default=False)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=175)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--rotate', type=float,
                    default=0, help='rotation')
parser.add_argument('--gblur', action='store_true', default=True)
parser.add_argument('--bblur', action='store_true', default=False)
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=1)
parser.add_argument('--bs_mult_val', type=int, default=0)
parser.add_argument('--crop_size', type=int, default=520,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt')
parser.add_argument('--tb_path', type=str, default='logs/tb')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Synchronized BN')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (1 epoch run ) to verify nothing failed')
parser.add_argument('--mode',type=str,default="train")
parser.add_argument('--test_sv_path', type=str, default="")
parser.add_argument('--checkpoint_path',type=str,default="")
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)
parser.add_argument('--maxSkip', type=int, default=0)
args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                        'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

#Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1
#Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

def load_image(image_file_path, transform=None):
    img = Image.open(image_file_path).convert('RGB')
    img = img.resize([520,520], Image.LANCZOS)

    if transform is not None:
        img = transform(img).unsqueeze(0)

    return img

# Image preprocessing
mean_std = ((0.496588, 0.59493099, 0.53358843), (0.496588, 0.59493099, 0.53358843))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.496588, 0.59493099, 0.53358843), (0.496588, 0.59493099, 0.53358843))])


def main():
    '''
    Main Function

    '''

    #Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    net = network.get_net_for_single_inference(args)
    #writer = prep_experiment(args,parser)
    #train_loader, val_loader, train_obj = datasets.setup_loaders(args)
    #criterion, criterion_val = loss.get_loss(args)
    #net = network.get_net(args, criterion)
    #optim, scheduler = optimizer.get_optimizer(args, net)


    torch.cuda.empty_cache()

    test_sv_path = args.test_sv_path
    print(f"Saving prediction {test_sv_path}")
    net.eval()

    print(f"Start the timer")
    start = time.time()

    test_image_path = '/home/tiga/Documents/IRP/RELLIS-3D/dataset/single_test/sample.jpg'
    test_img = load_image(test_image_path, transform)
    test_img_tensor = test_img.cuda()

    with torch.no_grad():
        seg_out, edge_out = net(test_img_tensor)    # output = (1, 19, 713, 713)

    semantic_mask = seg_out[0].max(0)[1].byte().cpu().data
    semantic_mask_color = Colorize()(semantic_mask.unsqueeze(0))

    save_path = os.path.join(test_sv_path,"gscnn","single_image_inference")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    semantic_mask_color_save = ToPILImage()(semantic_mask_color)
    semantic_mask_color_save.save(os.path.join(save_path,'mask.jpg'))

    print(f"Stop the timer")
    time_delta = time.time() - start
    print(f'Single Image inference finished in {time_delta // 60} mins {time_delta % 60} secs')

    return

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
