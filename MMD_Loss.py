import torch
import torch.nn as nn
# from numpy import *
import statistics
import cv2

bce_loss = nn.BCEWithLogitsLoss()

def dice_coeff(y_pred, y_true):
    smooth = 1.
    # Flatten
    y_true_f = torch.reshape(y_true, [-1])
    y_pred_f = torch.reshape(y_pred, [-1])
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score


def Dice_Loss(pred, y_true):
    y_pred = torch.sigmoid(pred)
    score = dice_coeff(y_pred, y_true)
    loss = 1 - score
    return loss

def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N

class SoftDiceLoss(nn.Module):
    def __init__(self, activation='sigmoid'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeffv2(y_pr, y_gt, activation=self.activation)

def Hyper_Loss(preds, masks, mode='mean'):
    loss = 0
    for pred, mask in zip(preds, masks):
        pred = pred.cuda()
        mask = mask.cuda()
        loss += bce_loss(pred, mask)
    if mode == 'mean':
        return loss / len(preds)
    elif mode == 'sum':
        return loss
    return loss

def DA_Loss(feats, mode='mean'):
    mmd_loss = MMDLoss()
    bs_loss = 0

    loss_mean = []
    bs = feats[0][0].shape[0]
    for id, feat in enumerate(feats):
        loss = 0
        for bat in range(0, bs):
            feats_t1, feats_t2 = feat[0][bat].flatten(1), feat[1][bat].flatten(1)
            loss += mmd_loss(feats_t1, feats_t2)
        loss_mean.append(loss / bs)
    # feats_t1, feats_t2 = feats[0][0][0].flatten(1), feats[0][1][0].flatten(1)
    # loss_mean = mmd_loss(feats_t1, feats_t2)
    if mode == 'mean':
        return sum(loss_mean) / len(loss_mean)
        # return loss_mean[4]
    elif mode == 'sum':
        # return loss_mean[4]
        return sum(loss_mean)
    elif mode == '0':
        return loss_mean[0]
    elif mode == '1':
        return loss_mean[1]
    elif mode == '2':
        return loss_mean[2]
    elif mode == '3':
        return loss_mean[3]
    elif mode == '4':
        return loss_mean[4]

class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss