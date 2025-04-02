# -*- coding: utf-8 -*-
"""
@author: stx
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from ..metrics.visualization_metrics import visualization
# from .shiyan2 import LinearCoModel
from .torchlinear import LinearCoModel
import torch.optim as optim


class GRU(nn.Module):
    """
    GRU模型
    """
    def __init__(self, num_series, hidden):
        super(GRU, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch):
        # Initialize hidden states
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)

    def forward(self, X, z, connection, mode="train"):
        X = X[:, :, np.where(connection != 0)[0]]
        device = self.gru.weight_ih_l0.device
        tau = 0
        if mode == "train":
            X_right, hidden_out = self.gru(
                torch.cat((X[:, 0:1, :], X[:, 11:-1, :]), 1), z
            )
            X_right = self.linear(X_right)

            return X_right, hidden_out


class VRAE4E(nn.Module):
    def __init__(self, num_series, hidden):
        """
        补偿网络
        """
        super(VRAE4E, self).__init__()
        # self.device = torch.device('cuda')
        self.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

        self.p = num_series
        self.hidden = hidden

        self.gru_left = nn.GRU(num_series, hidden, batch_first=True)
        self.gru_left.flatten_parameters()

        self.fc_mu = nn.Linear(hidden, hidden)  # nn.Linear(hidden, 1)
        self.fc_std = nn.Linear(hidden, hidden)

        self.linear_hidden = nn.Linear(hidden, hidden)
        self.tanh = nn.Tanh()

        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, num_series)

    def init_hidden(self, batch):
        """Initialize hidden states for GRU cell."""
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)

    def forward(self, X, mode="train"):
        try:
            X = torch.cat((torch.zeros(X.shape, device=self.device)[:, 0:1, :], X), 1)
        except Exception as e:
            import pdb
            pdb.set_trace()
            print(e)
        if mode == "train":
            hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
            out, h_t = self.gru_left(X[:, 1:, :], hidden_0.detach())

            mu = self.fc_mu(h_t)
            log_var = self.fc_std(h_t)

            sigma = torch.exp(0.5 * log_var)
            z = torch.randn(size=mu.size())
            z = z.type_as(mu)
            z = mu + sigma * z
            z = self.tanh(self.linear_hidden(z))

            X_right, hidden_out = self.gru(X[:, :-1, :], z)

            pred = self.linear(X_right)

            return pred, log_var, mu
        if mode == "test":
            X_seq = torch.zeros(X[:, :1, :].shape).to(self.device)
            h_t = torch.randn(size=(1, X_seq[:, -2:-1, :].size(0), self.hidden)).to(
                self.device
            )
            for i in range(int(20 / 1) + 1):
                out, h_t = self.gru(X_seq[:, -1:, :], h_t)
                out = self.linear(out)
                # out = self.sigmoid(out)
                X_seq = torch.cat([X_seq, out], dim=1)
            return X_seq


class CRVAE(nn.Module):
    def __init__(self, num_series, connection, hidden):
        """
        主干网络：循环神经网络
        """
        super(CRVAE, self).__init__()

        # self.device = torch.device('cuda')
        self.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

        self.p = num_series
        self.hidden = hidden

        self.gru_left = nn.GRU(num_series, hidden, batch_first=True)
        self.gru_left.flatten_parameters()

        self.fc_mu = nn.Linear(hidden, hidden)
        self.fc_std = nn.Linear(hidden, hidden)
        self.connection = connection

        # Set up networks.
        self.networks = nn.ModuleList(
            [GRU(int(connection[:, i].sum()), hidden) for i in range(num_series)]
        )
    
    def forward(self, X, noise=None, mode="train", phase=0):
        if phase == 0:
            X = torch.cat((torch.zeros(X.shape, device=self.device)[:, 0:1, :], X), 1)
            if mode == "train":
                hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
                out, h_t = self.gru_left(X[:, 1:11, :], hidden_0.detach())

                mu = self.fc_mu(h_t)
                log_var = self.fc_std(h_t)

                sigma = torch.exp(0.5 * log_var)
                z = torch.randn(size=mu.size())
                z = z.type_as(mu)
                z = mu + sigma * z

                pred = [
                    self.networks[i](X, z, self.connection[:, i])[0]
                    for i in range(self.p)
                ]

                return pred, log_var, mu
            if mode == "test":
                X_seq = torch.zeros(X[:, :1, :].shape).to(self.device)
                h_0 = torch.randn(size=(1, X_seq[:, -2:-1, :].size(0), self.hidden)).to(
                    self.device
                )
                ht_last = []
                for i in range(self.p):
                    ht_last.append(h_0)
                for i in range(int(20 / 1) + 1):  # int(20/2)+1
                    ht_new = []
                    for j in range(self.p):
                        # out, h_t = self.gru_out[j](X_seq[:,-1:,:], ht_last[j])
                        # out = self.fc[j](out)
                        out, h_t = self.networks[j](
                            X_seq[:, -1:, :], ht_last[j], self.connection[:, j]
                        )
                        if j == 0:
                            X_t = out
                        else:
                            X_t = torch.cat((X_t, out), -1)
                        ht_new.append(h_t)
                    ht_last = ht_new
                    if i == 0:
                        X_seq = X_t
                    else:
                        X_seq = torch.cat([X_seq, X_t], dim=1)

                    # out = self.sigmoid(out)

                return X_seq

        if phase == 1:
            X = torch.cat((torch.zeros(X.shape, device=self.device)[:, 0:1, :], X), 1)
            if mode == "train":
                hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
                out, h_t = self.gru_left(X[:, 1:11, :], hidden_0.detach())

                mu = self.fc_mu(h_t)
                log_var = self.fc_std(h_t)

                sigma = torch.exp(0.5 * log_var)
                z = torch.randn(size=mu.size())
                z = z.type_as(mu)  # Setting z to be .cuda when using GPU training
                z = mu + sigma * z

                pred = [
                    self.networks[i](X, z, self.connection[:, i])[0]
                    for i in range(self.p)
                ]

                return pred, log_var, mu
            if mode == "test":
                X_seq = torch.zeros(X[:, :1, :].shape).to(self.device)
                h_0 = torch.randn(size=(1, X_seq[:, -2:-1, :].size(0), self.hidden)).to(
                    self.device
                )
                ht_last = []
                for i in range(self.p):
                    ht_last.append(h_0)
                for i in range(int(20 / 1) + 1):  # int(20/2)+1
                    ht_new = []
                    for j in range(self.p):
                        # out, h_t = self.gru_out[j](X_seq[:,-1:,:], ht_last[j])
                        # out = self.fc[j](out)
                        out, h_t = self.networks[j](
                            X_seq[:, -1:, :], ht_last[j], self.connection[:, j]
                        )
                        if j == 0:
                            X_t = out
                        else:
                            X_t = torch.cat((X_t, out), -1)
                        ht_new.append(h_t)
                    ht_last = ht_new
                    if i == 0:
                        X_seq = X_t + 0.1 * noise[:, i : i + 1, :]
                    else:
                        X_seq = torch.cat(
                            [X_seq, X_t + 0.1 * noise[:, i : i + 1, :]], dim=1
                        )

                    # out = self.sigmoid(out)

                return X_seq

    def GC(self, threshold=True):
        """
        从多头解码器中提取初步的因果图

        Args:
          threshold: return norm of weights, or whether norm is nonzero.

        Returns:
          GC: (p x p) 的邻接矩阵. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        """
        GC = [torch.norm(net.gru.weight_ih_l0, dim=0) for net in self.networks]
        GC = torch.stack(GC)
        # print(GC)
        if threshold:
            # return (torch.abs(GC) > 0.3).int()
            return torch.abs(GC)
        else:
            return GC

    def GC_gai(self, threshold=True):
        """
        加入模糊残余因素后改进因果图

        Args:
          threshold: return norm of weights, or whether norm is nonzero.

        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        """
        # import pdb
        # pdb.set_trace()
        GC = [torch.norm(net.gru.weight_ih_l0, dim=0) for net in self.networks]
        # import pdb
        # pdb.set_trace()
        # GC = [g.unsqueeze(0) for g in GC]
        connection = self.connection.copy()
        try:
            GC = torch.stack(GC)
        except:
            for i in range(connection.shape[1]):
                # print(self.connection[:, i].sum())
                # print(GC[i].shape)
                for j in GC[i]:
                    # print(j)
                    for k in connection[:, i]:
                        if connection[k, i] == 1:
                            connection[k, i] = j.item()
                            j = j + 1

            GC = torch.tensor(connection)
        # print(GC.shape)
        # print(GC)
        if threshold:
            return (torch.abs(GC) > 0.3).int()
            # return torch.abs(GC)
        else:
            return GC


def prox_update(network, lam, lr):
    """对网络的第一层权重矩阵执行近端更新（原地操作）

    参数:
        network: 目标神经网络，需要包含GRU层
        lam (float): 正则化系数（控制稀疏性的超参数）
        lr (float): 学习率（控制更新步长的超参数）

    输出:
        无返回值，直接原地修改network.gru.weight_ih_l0的权重矩阵

    操作说明:
        1. 对权重矩阵的每列计算L2范数
        2. 通过软阈值函数进行收缩操作（当列范数 < lr*lam 时置零）
        3. 保持原始方向的同时缩放列向量
        4. 最后调用flatten_parameters保证参数连续性
    """
    W = network.gru.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = (W / torch.clamp(norm, min=(lam * lr))) * torch.clamp(
        norm - (lr * lam), min=0.0
    )
    network.gru.flatten_parameters()


def regularize(network, lam):
    """
    计算网络第一层权重矩阵的正则化项（L2/L1混合正则）

       参数:
           network: 目标神经网络，需要包含GRU层
           lam (float): 正则化系数（控制正则化强度的超参数）

       返回:
           torch.Tensor: 标量正则化损失值，计算公式为 lam * sum(||W[:,j]||_2)

       数学形式:
           相当于 Group Lasso 正则化，对权重矩阵的每列求L2范数后求和，
           可以促进列级别的稀疏性（整列趋于零）
    """
    W = network.gru.weight_ih_l0
    return lam * torch.sum(torch.norm(W, dim=0))


def ridge_regularize(network, lam):
    """Apply ridge penalty at linear layer and hidden-hidden weights."""
    return lam * (
        torch.sum(network.linear.weight**2) + torch.sum(network.gru.weight_hh_l0**2)
    )  # +
    # torch.sum(network.fc_std.weight ** 2) +
    # torch.sum(network.fc_mu.weight ** 2) +
    # torch.sum(network.fc_std.weight ** 2))


def restore_parameters(model, best_model):
    """计算网络线性层和GRU隐藏层权重的L2正则化项（岭回归惩罚项）

    参数:
        network: 目标神经网络，需要包含linear层和gru层
        lam (float): 正则化系数（控制L2惩罚强度的超参数）

    返回:
        torch.Tensor: 标量正则化损失值，计算方式为：
            lam * (||linear.weight||_F^2 + ||gru.weight_hh_l0||_F^2)

    数学形式:
        相当于对两个权重矩阵的Frobenius范数平方和进行惩罚：
        loss = λ*(∑_i,j W_linear[i,j]^2 + ∑_k,l W_gru_hh[k,l]^2)

    作用:
        1. 防止线性层和GRU隐藏层权重过大
        2. 缓解过拟合（经典岭回归/Tikhonov正则化）
        3. 保持权重平滑性
    """
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def arrange_input(data, context):
    """将单条时间序列切分成重叠的短序列（滑动窗口处理）

    参数:
        data (Tensor): 原始时间序列，形状为 (时间步长T, 特征维度dim)
        context (int): 滑动窗口长度（每个子序列的时间步数）

    返回:
        tuple: (input_sequences, target_sequences) 元组，其中:
            - input_sequences: 输入序列组，形状为 (批数量, 窗口长度, 特征维度)
            - target_sequences: 目标序列组，形状同输入序列组

    功能说明:
        1. 使用滑动窗口方法生成T-context组连续子序列
        2. 每组子序列中：
           - 输入序列 = 从t到t+context-1时刻的数据
           - 目标序列 = 从t+1到t+context时刻的数据（即输入序列的下一时刻）
        3. 适用于时间序列预测任务的训练数据准备
    """
    assert context >= 1 and isinstance(context, int)
    input = torch.zeros(
        len(data) - context,
        context,
        data.shape[1],
        dtype=torch.float32,
        device=data.device,
    )
    target = torch.zeros(
        len(data) - context,
        context,
        data.shape[1],
        dtype=torch.float32,
        device=data.device,
    )
    for i in range(context):
        start = i
        end = len(data) - context + i
        input[:, i, :] = data[start:end]
        target[:, i, :] = data[start + 1 : end + 1]
    return input.detach(), target.detach()


def MinMaxScaler(data):
    """Min-Max归一化处理器（将数据线性变换到[0,1]范围）

    参数:
        data (ndarray): 原始输入数据，支持任意维度的numpy数组
                       典型形状如：(样本数, 时间步长, 特征维度) 或 (样本数, 特征维度)

    返回:
        tuple: 包含三个元素的元组:
            - norm_data (ndarray): 归一化后的数据，与输入形状相同
            - min_val (ndarray): 各特征列的最小值向量，形状为(特征维度,)
            - max_val (ndarray): 各特征列的最大值向量，形状为(特征维度,)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data


def train_phase2(
    crvae,
    vrae,
    X,
    context,
    lr,
    max_iter,
    lam=0,
    lam_ridge=0,
    lookback=5,
    check_every=50,
    verbose=1,
    sparsity=100,
    batch_size=1024,
):
    """添加补偿网络的训练流程：联合优化CRVAE和VRAE模型
    
    参数:
        crvae (CRVAE): 因果循环变分自编码器模型
        vrae (VRAE): 补偿网络自编码器模型（用于误差建模）
        X (list): 输入数据列表，每个元素为[样本数, 时间步长, 特征维度]的张量
        context (int): 上下文窗口长度
        lr (float): 学习率
        max_iter (int): 最大迭代次数
        lam (float): 稀疏正则化系数（默认0）
        lam_ridge (float): L2正则化系数（默认0）
        lookback (int): 早停观察窗口（默认5）
        check_every (int): 验证间隔步数（默认50）
        verbose (int): 日志详细程度（0/1）
        sparsity (int): 稀疏性目标百分比（默认100）
        batch_size (int): 批大小（默认1024）
        
    返回:
        list: 训练损失记录

    """
    optimizer = optim.Adam(vrae.parameters(), lr=1e-3)
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    batch_size = batch_size
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)

    idx = np.random.randint(len(X_all), size=(batch_size,))

    X = X_all[idx]

    Y = Y_all[idx]
    X_v = X_all[batch_size:]
    start_point = 0  # context-10-1
    beta = 1  # 0.001
    beta_e = 1
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    # Calculate smooth error.
    pred, mu, log_var = crvae(X)  #

    loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

    mmd = (
        -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=-1).sum(dim=0)
    ).mean(dim=0)
    # mmd =  sum([MMD(torch.randn(200, Y[:, :, 0].shape[-1], requires_grad = False).to(device), latent[i][:,:,0]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
    smooth = loss + ridge + beta * mmd

    error = (-torch.stack(pred)[:, :, :, 0].permute(1, 2, 0) + X[:, 10:, :]).detach()
    pred_e, mu_e, log_var_e = vrae(error)
    loss_e = loss_fn(pred_e, error)
    mmd_e = (
        -0.5 * (1 + log_var_e - mu_e**2 - torch.exp(log_var_e)).sum(dim=-1).sum(dim=0)
    ).mean(dim=0)
    smooth_e = loss_e + beta_e * mmd_e

    best_mmd = np.inf

    ########################################################################
    # lr = 1e-3
    for it in range(max_iter):
        # Take gradient step.
        smooth_e.backward()
        if lam == 0:
            optimizer.step()
            optimizer.zero_grad()

        smooth.backward()
        for param in crvae.parameters():
            param.data -= lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in crvae.networks:
                prox_update(net, lam, lr)

        crvae.zero_grad()

        # Calculate loss for next iteration.
        idx = np.random.randint(len(X_all), size=(batch_size,))

        # X = X_all[idx]

        # Y = Y_all[idx]

        pred, mu, log_var = crvae(X)  #
        loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

        mmd = (
            -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=-1).sum(dim=0)
        ).mean(dim=0)

        ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
        smooth = loss + ridge + beta * mmd

        error = (
            -torch.stack(pred)[:, :, :, 0].permute(1, 2, 0) + X[:, 10:, :]
        ).detach()
        pred_e, mu_e, log_var_e = vrae(error)
        loss_e = loss_fn(pred_e, error)
        mmd_e = (
            -0.5
            * (1 + log_var_e - mu_e**2 - torch.exp(log_var_e)).sum(dim=-1).sum(dim=0)
        ).mean(dim=0)
        smooth_e = loss_e + beta_e * mmd_e

        # Check progress.
        if (it) % check_every == 0:
            X_t = X
            pred_t, mu_t, log_var_t = crvae(X_t)

            loss_t = sum(
                [loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)]
            )

            mmd_t = (
                -0.5
                * (1 + log_var_t - mu_t**2 - torch.exp(log_var_t))
                .sum(dim=-1)
                .sum(dim=0)
            ).mean(dim=0)

            ridge_t = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
            smooth_t = loss_t + ridge_t  # + beta*mmd_t

            nonsmooth = sum([regularize(net, lam) for net in crvae.networks])
            mean_loss = (smooth_t) / p

            if verbose > 0:
                print(("-" * 10 + "Iter = %d" + "-" * 10) % (it))
                print("Loss = %f" % mean_loss)
                print("KL = %f" % mmd)

                print("Loss_e = %f" % smooth_e)
                print("KL_e = %f" % mmd_e)

                if lam > 0:
                    print(
                        "Variable usage = %.2f%%"
                        % (100 * torch.mean(crvae.GC().float()))
                    )

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crvae)

            start_point = 0
            predicted_error = vrae(error, mode="test").detach()

            predicted_data = crvae(X_t, predicted_error, mode="test", phase=1)
            syn = predicted_data[:, :-1, :].cpu().detach().numpy()
            ori = X_t[:, start_point:, :].cpu().detach().numpy()

            if it % 1000 == 0:
                plt.plot(ori[0, :, 1])
                plt.plot(syn[0, :, 1])
                plt.show()

                visualization(ori, syn, "pca")
                visualization(ori, syn, "tsne")
                np.save("ori_henon.npy", ori)
                np.save("syn_henon.npy", syn)

    # Restore best model.
    restore_parameters(crvae, best_model)

    return train_loss_list


def cal_loss_batched(Xs, connections):
    batch_size, n, d = Xs.shape
    Id = np.eye(d).astype(np.float64)
    Ids = np.broadcast_to(Id, (batch_size, d, d))  # 广播到每个批次

    covs = np.einsum('bni,bnj->bij', Xs, Xs) / n
    difs = Ids - connections
    rhs = np.einsum('bij,bjk->bik', covs, difs)
    losses = 0.5 * np.einsum('bij,bji->b', difs, rhs) / (d * d)
    G_losses = -rhs

    return losses, G_losses



def train_phase1(crvae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                 lookback=5, check_every=50, verbose=1, sparsity=100, batch_size=2048):
    '''基线方法：Train model with Adam.'''
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    batch_size = batch_size
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)

    idx = np.random.randint(len(X_all), size=(batch_size,))

    X = X_all[idx]

    Y = Y_all[idx]
    start_point = 0
    beta = 0.1
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    # Calculate crvae error.
    pred, mu, log_var = crvae(X)

    loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

    mmd = (-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=-1).sum(dim=0)).mean(dim=0)
    # mmd =  sum([MMD(torch.randn(200, Y[:, :, 0].shape[-1], requires_grad = False).to(device), latent[i][:,:,0]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
    smooth = loss + ridge + beta * mmd

    best_mmd = np.inf

    ########################################################################
    # lr = 1e-3
    for it in range(max_iter):
        # Take gradient step.
        smooth.backward()
        for param in crvae.parameters():
            param.data -= lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in crvae.networks:
                prox_update(net, lam, lr)

        crvae.zero_grad()

        pred, mu, log_var = crvae(X)
        loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

        mmd = (-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=-1).sum(dim=0)).mean(dim=0)

        ridge = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
        smooth = loss + ridge + beta * mmd

        # Check progress.
        if (it) % check_every == 0:
            X_t = X
            Y_t = Y

            pred_t, mu_t, log_var_t = crvae(X_t)

            loss_t = sum([loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)])

            mmd_t = (-0.5 * (1 + log_var_t - mu_t ** 2 - torch.exp(log_var_t)).sum(dim=-1).sum(dim=0)).mean(dim=0)

            ridge_t = sum([ridge_regularize(net, lam_ridge)
                           for net in crvae.networks])
            smooth_t = loss_t + ridge_t  # + beta*mmd_t

            nonsmooth = sum([regularize(net, lam) for net in crvae.networks])
            mean_loss = (smooth_t) / p

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it))
                print('Loss = %f' % mean_loss)
                print('KL = %f' % mmd)

                if lam > 0:
                    print('Variable usage = %.2f%%'
                          % (100 * torch.mean(crvae.GC().float())))

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crvae)

            start_point = 0
            predicted_data = crvae(X_t, mode='test')
            syn = predicted_data[:, :-1, :].cpu().detach().numpy()
            ori = X_t[:, start_point:, :].cpu().detach().numpy()

            syn = MinMaxScaler(syn)
            ori = MinMaxScaler(ori)

    # Restore best model.
    restore_parameters(crvae, best_model)

    return train_loss_list



def cal_loss_batched_torch(Xs, connections,device='cuda:0'):
    batch_size, n, d = Xs.shape
    Id = torch.eye(d, dtype=torch.float32).to(device)
    Ids = Id.expand(batch_size, d, d)  # 使用expand来广播到每个批次

    covs = torch.einsum('bni,bnj->bij', Xs, Xs) / n
    difs = Ids - connections
    rhs = torch.einsum('bij,bjk->bik', covs, difs)
    losses = 0.5 * torch.einsum('bij,bji->b', difs, rhs) / (d * d)
    G_losses = -rhs

    return losses, G_losses



def train_phase3(
    crvae,
    X,
    context,
    lr,
    max_iter,
    lam=0,
    lam_ridge=0,
    lookback=5,
    check_every=50,
    verbose=1,
    sparsity=100,
    batch_size=2048,
):
    """带边评分函数的CRVAE优化

    参数:
        crvae (CRVAE): 因果循环变分自编码器模型
        X (list): 输入数据列表，每个元素为[样本数, 时间步长, 特征维度]的张量  
        context (int): 上下文窗口长度
        lr (float): 学习率
        max_iter (int): 最大迭代次数
        lam (float): 稀疏正则化系数（默认0）
        lam_ridge (float): L2正则化系数（默认0）
        lookback (int): 早停观察窗口（默认5）
        check_every (int): 验证间隔步数（默认50）
        verbose (int): 日志详细程度（0/1）
        sparsity (int): 稀疏性目标百分比（默认100）
        batch_size (int): 批大小（默认2048）

    返回:
        list: 训练损失记录

    算法流程:
        1. 数据准备：滑动窗口处理
        2. 四部分损失计算：
           - 重构损失（MSE）
           - KL散度（正则化潜在空间） 
           - 岭回归（L2正则）
           - 因果约束损失（评分函数）
        3. 梯度下降优化
        4. 近端算子处理稀疏性
        5. 早停机制保留最佳模型
    """
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device

    loss_fn = nn.MSELoss()
    train_loss_list = []
    batch_size = batch_size
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)

    idx = np.random.randint(len(X_all), size=(batch_size,))

    X = X_all[idx]

    Y = Y_all[idx]


    start_point = 0
    beta = 0.1
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    # Calculate crvae error.
    pred, mu, log_var = crvae(X)

    loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

    mmd = (
        -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=-1).sum(dim=0)
    ).mean(dim=0)
    # mmd =  sum([MMD(torch.randn(200, Y[:, :, 0].shape[-1], requires_grad = False).to(device), latent[i][:,:,0]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])

    # liner_loss = 0
    # for x in X:
    #     liner_loss += cal_loss(x.cpu().numpy(),crvae.connection)[0]
    # liner_loss /= len(X)

    X_np = X.cpu().numpy()  # X 是原始的 PyTorch tensor
    h_model = LinearCoModel(loss_type='l2', lambda1=0.1,device='cuda')
    s_1=0.1
    mu_1 = 0.5
    # import pdb 
    # pdb.set_trace()

    losses,G_grad = h_model.integrated_loss(pred,torch.tensor(crvae.connection,device=device,dtype=torch.float32), mu_1,s_1)
    # print(X_np,X_np.shape)
    # print(crvae.connection.shape)
    # # 计算所有样本的损失和梯度
    # losses, G_losses = cal_loss_batched(X_np, crvae.connection)

    # 计算平均损失
    # average_loss = np.mean(losses)
    gamma = 10
    # smooth = loss + ridge + beta * mmd
    smooth = loss + ridge + beta * mmd + gamma * losses

    best_mmd = np.inf

    ########################################################################
    # lr = 1e-3
    for it in range(max_iter):
        # Take gradient step.
        smooth.backward()
        for param in crvae.parameters():
            # print("param.grad",param.grad)
            # print("param.grad.shape",param.grad.shape)
            param.data -= lr * param.grad



        # Take prox step.
        if lam > 0:
            for net in crvae.networks:
                prox_update(net, lam, lr)

        crvae.zero_grad()

        pred, mu, log_var = crvae(X)
        loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

        mmd = (
            -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=-1).sum(dim=0)
        ).mean(dim=0)

        ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
        losses,G_grad = h_model.integrated_loss(pred,torch.tensor(crvae.connection,device=device,dtype=torch.float32), mu_1,s_1)
        smooth = loss + ridge + beta * mmd + gamma * losses

        # Check progress.
        if (it) % check_every == 0:
            X_t = X
            Y_t = Y

            pred_t, mu_t, log_var_t = crvae(X_t)

            loss_t = sum(
                [loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)]
            )

            mmd_t = (
                -0.5
                * (1 + log_var_t - mu_t**2 - torch.exp(log_var_t))
                .sum(dim=-1)
                .sum(dim=0)
            ).mean(dim=0)

            ridge_t = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])

            losses_t, G_grad_t = h_model.integrated_loss(pred,torch.tensor(crvae.connection,device=device,dtype=torch.float32), mu_1, s_1)
            smooth_t = loss_t + ridge_t + gamma * losses_t # + beta*mmd_t

            nonsmooth = sum([regularize(net, lam) for net in crvae.networks])
            mean_loss = (smooth_t) / p

            if verbose > 0:
                print(("-" * 10 + "Iter = %d" + "-" * 10) % (it))
                print("Losses = %f" % losses_t)
                print("Loss = %f" % mean_loss)
                print("KL = %f" % mmd)

                if lam > 0:
                    print(
                        "Variable usage = %.2f%%"
                        % (100 * torch.mean(crvae.GC().float()))
                    )

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crvae)

            start_point = 0
            predicted_data = crvae(X_t, mode="test")
            syn = predicted_data[:, :-1, :].cpu().detach().numpy()
            ori = X_t[:, start_point:, :].cpu().detach().numpy()

            syn = MinMaxScaler(syn)
            ori = MinMaxScaler(ori)


    # Restore best model.
    restore_parameters(crvae, best_model)

    return train_loss_list


