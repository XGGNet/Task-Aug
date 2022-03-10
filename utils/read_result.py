# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:55:41 2020

@author: Sunly
"""

import numpy as np
import argparse
import re
import tensorflow as tf


# def get_args():
#     parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("--root", '--p',type=str,default='./result_paper/')
#     # parser.add_argument("--repeat", '--r',type=int, default=0)
#     # parser.add_argument("--method", '--m', type=str, default='Rakelly')#'cons1_ms_w_layer9_t0.0_margin0_pretrain')
#     return parser.parse_args()

# args = get_args()

root = './log'
dataset = ['VGH','NKI','IHC','NCH']
repeat = ['run1','run2','run3']
expname = ''

acc_list = []
for data in dataset:
    folder = f'{log}/mame_{data}/{expname}/'
    _acc_list = []
    for pt in repeat:
        model_file = tf.train.latest_checkpoint(folder + f'checkpoint_{pt}')
        acc = float(model_file.split('acc')[-1])
        _acc_list.append(acc)
    mean = np.mean(_acc_list)
    std = np.std(_acc_list,ppof=1)
    print(f'exp is {expname}, target_set is {data}, acc={acc[0]:.5f},{acc[1]:.5f},{acc[2]:.5f}, crossrun_mean={mean:.5f}, std={std:.5f}')
    acc_list.append(mean)
acc_mean = np.mean(_acc_list)
print(f'****exp is {expname}, crossset_mean={acc_mean:.5f})


















































