import torch
import numpy as np

import time
import torch.nn as nn
from torch.nn import init
import os
import sys
import cv2
from utils.model_init import model_init
from framework import Framework
from utils.datasets import prepare_PETCT_dataset
#from networks.petctnet import PETCT
from networks.linknet34_fuse import LinkNet34_fuse

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def get_model(model_name):
    if model_name == 'CMMPNet':
       model = LinkNet34_fuse()
    else:
        print("[ERROR] can not find model ", model_name)
        assert(False)
    return model

def get_dataloader(args):
    train_ds, val_ds, test_ds = prepare_PETCT_dataset(args)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=True,  drop_last=False)
    val_dl   = torch.utils.data.DataLoader(test_ds,   batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False, drop_last=False)
    test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False, drop_last=False)
    return train_dl, val_dl, test_dl


def train_val_test(args):
    net = get_model(args.model)
    #print(net)

    print('lr:',args.lr)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
#     new_state = {}
#     net.load_state_dict(new_state)

    framework = Framework(net, optimizer, dataset=args.dataset)
    
    train_dl, val_dl, test_dl = get_dataloader(args)
    framework.set_train_dl(train_dl)
    framework.set_validation_dl(val_dl)
    framework.set_test_dl(test_dl)
    framework.set_save_path(WEIGHT_SAVE_DIR)

    framework.fit(cos_lr=args.cos_lr,lam=args.lam,t=args.t,epochs=args.epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CMMPNet')
    parser.add_argument('--lr',    type=float, default=2e-4)
    parser.add_argument('--name',  type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pet_dir', type=str, default='E:/ML_data/PCLC1-108/PET/')
    parser.add_argument('--mask_dir', type=str, default='E:/ML_data/PCLC1-108/CT/')
    parser.add_argument('--ct_dir', type=str, default='E:/ML_data/PCLC1-108/CT/')

    parser.add_argument('--split_train_val_test', type=str, default='')
    parser.add_argument('--weight_save_dir', type=str, default='./save_model')
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--use_gpu',  type=bool, default=True)
    parser.add_argument('--gpu_ids',  type=str, default='0')
    parser.add_argument('--workers',  type=int, default=0)
    parser.add_argument('--epochs',  type=int, default=30)
    parser.add_argument('--lam', type=float, default=3e-4)
    parser.add_argument('--t', type=float, default=0.6)
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='BJRoad')
    parser.add_argument('--down_scale', type=bool, default=True)
    parser.add_argument('--cos_lr', type=bool, default=False)
    args = parser.parse_args()

    if args.use_gpu:
        try:
            gpu_list = [int(s) for s in args.gpu_ids.split(',')]
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        BATCH_SIZE = args.batch_size * len(gpu_list)
    else:
        BATCH_SIZE = args.batch_size
        
    WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir, f"{args.model}_{args.dataset}_"+time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())+"/")
    if not os.path.exists(WEIGHT_SAVE_DIR):
        os.makedirs(WEIGHT_SAVE_DIR)
    print("Log dir: ", WEIGHT_SAVE_DIR)
    
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(WEIGHT_SAVE_DIR+'train.log')

    train_val_test(args)
    print("[DONE] finished")

