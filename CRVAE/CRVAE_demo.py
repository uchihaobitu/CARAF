# -*- coding: utf-8 -*-
"""
@author: stx
"""

import os
import glob


import numpy as np
import torch
import os

from os.path import join, dirname, basename
from .models.cgru_error import CRVAE, VRAE4E, train_phase1,train_phase2,train_phase3

def cause_trian(data_path,output_dir,device):

    X_np = np.load(data_path)
    data_dir = os.path.dirname(data_path)
    print('data_dir',data_dir)
    service, metric = basename(dirname(dirname(data_path))).split("_")
    number = os.path.basename(data_dir)
    base_filename = os.path.join(output_dir, f"{service}_{metric}_{number}")
    print('base_filename',base_filename)
    X_np = X_np.astype(np.float32)

    print(X_np.shape)
    print(X_np)

    dim = X_np.shape[-1]
    GC = np.zeros([dim, dim])
    for i in range(dim):
        GC[i, i] = 1
        if i != 0:
            GC[i, i - 1] = 1
    X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)


    full_connect = np.ones(GC.shape)
    # full_connect = np.load('W_est.npy')
    # full_connect = torch.tensor(full_connect,device=device,dtype=torch.float64)
    print(full_connect.shape)
    cgru = CRVAE(X.shape[-1], full_connect, hidden=64).to(device=device)
    vrae = VRAE4E(X.shape[-1], hidden=64).to(device=device)



    # 基线方法
    # train_loss_list = train_phase1(
    #     cgru, X, context=20, lam=0.1, lam_ridge=0, lr=5e-2, max_iter=1000, check_every=50, batch_size=32
    # )  # 0.1

    # 加上评分函数
    train_loss_list = train_phase3(
        cgru, X, context=20, lam=0.1, lam_ridge=0, lr=5e-2, max_iter=1000, check_every=50, batch_size=128
    )  # 0.1


    # %%no
    GC_est = cgru.GC().cpu().data.numpy()
    cor_values = GC_est.flatten()
    # 计算保留的边数：总数的15%
    num_edges_to_keep = int(len(cor_values) * 0.05)

    # 找到前5%的最小值，只有大于或等于这个值的元素会被设置为1
    threshold = np.sort(cor_values)[-num_edges_to_keep]
    print('threshold',threshold)

    # 使用阈值更新原始矩阵：大于等于阈值的设置为1，其他设置为0
    result_matrix = np.where(GC_est >= threshold, 1,0)
    np.save(f'{base_filename}.npy', result_matrix)

    full_connect = np.load(f'{base_filename}.npy')

    cgru = CRVAE(X.shape[-1], full_connect, hidden=64).to(device=device)
    vrae = VRAE4E(X.shape[-1], hidden=64).cuda(device=device)

    #加上补偿网络

    train_loss_list = train_phase2(
        cgru, vrae, X, context=20, lam=0., lam_ridge=0, lr=5e-2, max_iter=1000,
        check_every=50,batch_size=128)
    GC_new = cgru.GC_gai().cpu().data.numpy()
    np.save(f'{base_filename}new.npy', GC_new)






