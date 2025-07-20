
from dataset1 import data_loder

import torch

from torch.optim.lr_scheduler import LambdaLR

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from step_by_step.informer_auto import SFETSMSPEDS_transformer

import matplotlib.pyplot as plt
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 让卷积结果可复现
    torch.backends.cudnn.benchmark = False     # 禁止自动优化
class SeasonalContrastLoss(torch.nn.Module):
    def __init__(self, mode='head_tail_quarter', reduction='mean'):
        """
        Args:
            metric: 'cosine' or 'l1'
            mode: 当前只支持 head_tail_quarter（首尾对齐相同季度）
            reduction: 'mean' or 'none'
        """
        super(SeasonalContrastLoss, self).__init__()

        self.mode = mode
        self.reduction = reduction

    def forward(self, feat, label):
        """
        Args:
            feat: [B, C, L] - 时间特征
            label: [B] - 每个样本是否发生变化（0 不变 / 1 变化）
        Returns:
            loss: scalar or [B] depending on reduction
        """
        B, C, L = feat.shape
        assert L >= 12, "建议时间序列为12的倍数（或至少一个完整周期）"

        # 设定季度起始位置（假设4季度，每季度3帧）
        quarter_ids = L//4  # Q1-Q4 start idx
        diffs = []

        for q in range(1,quarter_ids):

            f_q_head_1 = feat[:, :, 0]     # Qx 前部特征（如第1个月）

            f_q_end_1 = feat[:, :, -1]  # Qx 前部特征（如第1个月）

            f_q_endtail_1 = feat[:, :, L-4*q-1]   # Qx 后部特征（如第3个月）

            f_q_headtail_1 = feat[:, :, 4*q]  # Qx 后部特征（如第3个月）



            diff = torch.tanh(torch.abs(f_q_head_1 - f_q_headtail_1).mean(dim=1)+ torch.abs(f_q_end_1 - f_q_endtail_1).mean(dim=1))# [B]
            diffs.append(diff)

        all_diff = torch.stack(diffs, dim=1)  # [B, 4]
        mean_diff = all_diff.mean(dim=1)      # [B]
        print(mean_diff,'meandiff')


        # label ∈ {0, 1}，目标是差异对齐标签（不变=0，变化=1）
        target = label.float()  # [B]
        loss = F.mse_loss(mean_diff, target, reduction=self.reduction)


        return loss



def train():
    # train

    # 超参数
    EPOCH =15 # 前向后向传播迭代次数
    LR = 0.0001

    train7 = data_loder.load_buchong_dataset(32, 'datasetE/ll/city3_gaidata.txt', 'datasetE/ll/city3_gaitime.txt',
                                             'datasetE/ll/city3_gailabel.txt')
    Net=SFETSMSPEDS_transformer.ConvDisRNN(6,2).cuda()

    optimizer = torch.optim.Adam(Net.parameters(), betas=(0.9, 0.99), lr=LR, weight_decay=0.0001)

    scheduler_1 = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.86 ** epoch)
    sealoss=SeasonalContrastLoss()

    # 创建交叉熵损失函数
    criterion1 = nn.CrossEntropyLoss().cuda()

    LOSS=[]
    for epoch in range(EPOCH):
        Losssum=[]

        for step, data1 in enumerate(train7):

            x, x_mark, label = data1
            x = x.permute(0, 4, 1, 2, 3).cuda()
            x = x  # 加 1e-8 以防除零\
            x_mark = x_mark[:, 4, 4, :, :].cuda()
            time_len = x.shape[4]
            pred,out1,out2 = Net(x, x_mark, time_len)

            la = label[:, 4, 4].long().cuda()

            one_hot_targets = torch.nn.functional.one_hot(la, num_classes=2).float()
            loss = criterion1(pred, one_hot_targets)+2*F.mse_loss(out1,la.float())+sealoss(out2,la.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            Losssum.append(loss.data.cpu())
            print(epoch, 'Epoch: | train loss: %.6f' % loss.data, '|')


        scheduler_1.step()
        LOSS.append(sum(Losssum))
        torch.save(Net.state_dict(), 'md/seed_1SFETSMSPEDS_transformer.pt')


    plt.plot(LOSS, marker='o', linestyle='-', color='b')
    plt.show()


if __name__ == '__main__':
    # 在主程序中调用train
    set_seed(42)
    train()