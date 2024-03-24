# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# loss_function


# construction of blocks
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size):
        super().__init__()
        self.add_module("bn", nn.BatchNorm1d(in_ch))
        self.add_module("act", nn.ReLU())
        self.add_module("conv", nn.Conv1d(in_ch, out_ch, k_size, padding=k_size//2))

    def forward(self, X):
        for blk in self._modules.values():
            X = blk(X)
        return X


class DenseBlock(nn.Module):
    def __init__(self, in_ch, conv_num, ch_num):
        super().__init__()
        self._out_ch = in_ch + ch_num * conv_num
        for i in torch.arange(conv_num):
            self.add_module(f"basic_blk{i}", BasicBlock(in_ch, ch_num, 3))
            in_ch += ch_num

    def forward(self, X):
        for blk in self._modules.values():
            Y = blk(X)
            X = torch.cat([X, Y], dim=1)
        return X

    def out_ch(self):
        return self._out_ch


class TransBlock(nn.Module):
    def __init__(self, in_ch, conv_k_num, conv_k_size, pool_k_size, pool_step):
        super().__init__()
        self.add_module("basic_blk", BasicBlock(in_ch, conv_k_num, conv_k_size))
        self.add_module("pool", nn.MaxPool1d(pool_k_size, pool_step, padding=pool_k_size//2))

    def forward(self, X):
        for blk in self._modules.values():
            X = blk(X)
        return X

# the class of structure of the 1D-Conv-DensNet
class Conv_DenseNet_1D(nn.Module):

    device = torch.device('cuda:0')

    def __init__(self, bit_num):
        super().__init__()
        # 密连接部分
        self.bit_num = bit_num
        dense_part = nn.Sequential()
        dense_part.add_module("conv1", nn.Conv1d(2, 64, 3, 1, 1))
        dense_part.add_module("trans_blk1", TransBlock(64, 128, 3, 3, 2))
        dense_part.add_module("dense_blk1", DenseBlock(128, 2, 128))
        dense_part.add_module("trans_blk2", TransBlock(128 * 2 + 128, 64, 3, 3, 2))
        dense_part.add_module("dense_blk2", DenseBlock(64, 3, 64))
        dense_part.add_module("trans_blk3", TransBlock(64 * 3 + 64, 64, 3, 3, 2))
        dense_part.add_module("dense_blk3", DenseBlock(64, 4, 64))
        dense_part.add_module("trans_blk4", TransBlock(64 * 4 + 64, 64, 3, 3, 2))
        dense_part.add_module("dense_blk4", DenseBlock(64, 3, 64))
        dense_part.add_module("conv2", nn.Conv1d(64 * 3 + 64, 150, 3, 1, 1))
        self.add_module("dense_part", dense_part)
        # 全局池化部分
        pool_part = nn.Sequential()
        pool_part.add_module("max_pool", nn.AdaptiveMaxPool1d(1))
        pool_part.add_module("avg_pool", nn.AdaptiveAvgPool1d(1))
        self.add_module("global_pool_part", pool_part)

        # 全连接部分
        self.add_module("lin_part", nn.Sequential(nn.Linear(300, self.bit_num*2)))

    def init(self):
        for m in self.modules():
            if hasattr(m, "weight"):
                nn.init.xavier_normal_(m.weight.data.unsqueeze(0))
        self.to(self.device)

    def loss(self, Y_hat, y):
        return F.cross_entropy(Y_hat.reshape([-1, 2]), y.reshape([-1]))

    def forward(self, X):
        models = self._modules
        pool_part = models["global_pool_part"]
        Y = models["dense_part"](X)
        feature_vec = torch.cat([pool_part[0](Y), pool_part[1](Y)], dim=1)
        y = models["lin_part"](feature_vec.flatten(1))
        return y.reshape([-1, self.bit_num, 2])

    def predict(self, X):
        with torch.no_grad():
            self.eval()
            Y_hat = self(X)
            y_hat = torch.argmax(Y_hat, dim=Y_hat.dim()-1)
        return y_hat


def split_data(data_set, split_rate, shuffle=True):
    feature, label = data_set
    data_num = len(feature)
    index = torch.arange(data_num)
    if shuffle:
        random.shuffle(index)
    split_gap = np.array(split_rate).cumsum()
    split_point = (split_gap * data_num / split_gap[-1]).astype(int)
    index_start = [0, *split_point[:-1]]
    index_end = split_point
    data_parts = []
    for start, end in zip(index_start, index_end):
        i = index[start:end]
        data_parts.append((feature[i], label[i]))
    return data_parts


def load_data(data_set, batch_size, shuffle=False):
    feature, label = data_set

    # convert IQ sequence to sequence ((real_part_sequence),(imaginary_part_sequence))
    real_feature = np.concatenate([feature.real, feature.imag], axis=1)
    real_feature = real_feature.reshape([len(feature), 2, -1])
    real_feature = torch.tensor(real_feature, dtype=torch.float)
    label = torch.tensor(label)

    data_num = len(real_feature)
    indices = np.arange(data_num)

    if shuffle:
        random.shuffle(indices)

    for start in range(0, data_num, batch_size):
        end = min(start+batch_size, data_num)
        yield real_feature[indices[start:end]], label[indices[start:end]]


def run_epoch(net, data_set, batch_size, optimizer=None):
    err_bits = 0
    is_train = optimizer is not None
    for X, y in load_data(data_set, batch_size, is_train):
        X = X.to(net.device)
        y = y.to(net.device)
        if is_train:
            optimizer.zero_grad()
            Y_hat = net(X)
            loss = net.loss(Y_hat, y)
            loss.backward()
            with torch.no_grad():
                optimizer.step()
                y_hat = torch.argmax(Y_hat, dim=Y_hat.dim()-1)
        else:
            y_hat = net.predict(X)
        err_bits += torch.sum(~(y_hat == y))
    return err_bits.cpu() / (len(data_set[0]) * net.bit_num)


def train(net, data_set, batch_size, lr, lr_gain, epochs, steady_epochs, momentum):
    steady_epoch = 0
    train_ebr_log = []
    for epoch in range(epochs):
        train_set = data_set
        if steady_epoch >= steady_epochs:
            lr *= lr_gain
            steady_epoch = 0
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum)
        # 训练
        ebr_train = run_epoch(net, train_set, batch_size, optimizer)
        train_ebr_log.append(ebr_train)
        print(f"epoch{epoch+1},ebr:{ebr_train}")

        # 记一次lr稳定的epoch
        steady_epoch += 1
    return train_ebr_log

