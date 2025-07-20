import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .PatchTST import Model

import matplotlib.pyplot as plt
import seaborn as sns


from .PatchTST_layers import series_decomp
from .RevIN import RevIN
from itertools import combinations

def visualize_attention_heatmap(attn_c, attn_s, sample_idx=0):
    """
    可视化 Channel Attention 和 Spatial Attention 的热力图
    attn_c: [B, C, 1]
    attn_s: [B, 1, L]
    """

    # detach + squeeze
    attn_c = attn_c[sample_idx].squeeze().cpu().detach().numpy()  # [C]
    attn_s = attn_s[sample_idx].squeeze().cpu().detach().numpy()  # [L]

    plt.figure(figsize=(12, 5))

    # Channel Attention 热力图
    plt.subplot(1, 2, 1)
    sns.heatmap(attn_c[np.newaxis, :], cmap='YlGnBu', cbar=True, xticklabels=True, yticklabels=['Channel'])
    plt.title("Channel Attention")
    plt.xlabel("Channel Index")

    # Spatial Attention 热力图
    plt.subplot(1, 2, 2)
    sns.heatmap(attn_s[np.newaxis, :], cmap='YlOrRd', cbar=True, xticklabels=False, yticklabels=['Time'])
    plt.title("Spatial (Temporal) Attention")
    plt.xlabel("Time Step Index")

    plt.tight_layout()
    plt.show()

def temporal_group_cosine_difference(x, window=4):
    """
    基于余弦相似度计算 group 变化得分。

    输入:
        x: [B, C, G, L]
        window: 前后 group 的最大比较窗口

    输出:
        diff_scores: [B, C, G, L]，每个 group 与其邻居的相似度差异（越大越可能变化）
    """
    B, C, G, L = x.shape
    diff_scores = torch.zeros_like(x)

    for g in range(G):
        neighbor_feats = []
        for offset in range(1, window + 1):
            if g - offset >= 0:
                neighbor_feats.append(x[:, :, g - offset, :])  # [B, C, L]
            if g + offset < G:
                neighbor_feats.append(x[:, :, g + offset, :])  # [B, C, L]

        if not neighbor_feats:
            continue

        # [B, C, N, L]
        neighbors = torch.stack(neighbor_feats, dim=2)  # N 为邻居数
        cur = x[:, :, g, :].unsqueeze(2)   # [B, C, 1, L]

        dif=torch.mean(torch.abs(cur-neighbors),dim=2)


        diff_scores[:, :, g, :] =dif

    return diff_scores  # [B, C, G, L]

class SpectralAttentionFFN(nn.Module):
    def __init__(self,  dim, num_heads=3):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, L, G = x.shape

        x_reshaped = x.permute(0, 3, 2, 1).reshape(B*G,L, C)

        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        x_out = self.norm(x_reshaped + attn_out)
        ffn_out = self.ffn(x_out)
        x_out = x_out + ffn_out  # 残差连接
        x_out = x_out.reshape(B,G,L, C).permute(0, 3, 2, 1).contiguous()

        return x_out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):


        avg_out = self.avg_pool(x)  # 对平均池化的特征进行处理
        max_out = self.max_pool(x) # 对最大池化的特征进行处理
        out = avg_out
        att=self.sigmoid(out)
        return out,att


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv1d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        out = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        out = self.conv1(out)  # 通过卷积层处理连接后的特征图
        x = self.sigmoid(out)*x
        return x,out # 使用sigmoid激活函数计算注意力权重

class ConvDisRNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConvDisRNN, self).__init__()

        self.frnet = Model()
        self.cac = ChannelAttention(128, 4)  # 通道注意力实例
        self.sac = SpatialAttention()  # 空间注意力实例
        self.cat = ChannelAttention(6, 1)  # 通道注意力实例
        self.sat = SpatialAttention()  # 空间注意力实例

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels, 16, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),

        )


        self.conv2_drop = nn.Dropout2d(p=0.3)  # Dropout layer
        self.fc1 = nn.Linear(134, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(6, num_classes)
        self.m = nn.Softmax(dim=1)
        self.mutattn=SpectralAttentionFFN(6,3)

        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x,x_mark, time_len):

        t_out = []

        # 提取年份和月份
        year = x_mark[0, :, 0]  # 假设所有 batch 的年份一致，取第一个 batch
        month = x_mark[0, :, 1]  # 假设所有 batch 的月份一致，取第一个 batch

        quarter = (month - 1) // 3 + 1
        # 按 (year, month) 分组
        unique_keys = torch.unique(torch.stack([year, quarter], dim=1), dim=0)  # 获取唯一的 (year, month) 对
        avg_by_group = {}

        # 计算每个组的平均值
        for key in unique_keys:
            mask = (year == key[0]) & (quarter == key[1])  # 找到对应组的 mask
            avg_by_group[(key[0].item(), key[1].item())] = x[:, :, :, :, mask].mean(dim=4)  # [batch, feature]

        # 计算相邻年份的差分
        keys_sorted = sorted(avg_by_group.keys(), key=lambda x: (x[1], x[0]))  # 按 (year, month) 排序
        diffs = {}
        for i in range(1, len(keys_sorted)):
            prev_key = keys_sorted[i - 1]
            curr_key = keys_sorted[i]
            # 确保是相邻年份且相同月份
            if curr_key[1] == prev_key[1] and curr_key[0] == prev_key[0] + 1:
                diffs[curr_key] = avg_by_group[curr_key] - avg_by_group[prev_key]

        diff_keys_sorted = sorted(diffs.keys())  # 确保差分结果按时间顺序
        t_stack = torch.stack([diffs[key] for key in diff_keys_sorted], dim=-1)  # 在最后一维堆叠

        for i in range(t_stack.shape[4]):
            t = self.conv2(self.conv1(t_stack[..., i]))
            t = self.conv5(self.conv4(self.conv3(t) + t))
            t = self.conv8(self.conv7(self.conv6(t) + t))
            t = self.conv10(self.conv9(t) + t)
            t = t[:, :, 1, 1]
            t_out.append(t)
        t_out = torch.stack(t_out, dim=-1)
        out1=torch.tanh(torch.abs(t_out).mean(dim=(1, 2)))
        t_out, att2 = self.cac(t_out)
        t_out = t_out.squeeze(2)


        data = x
        normalized_tensor=data[:,:,4,4,:].cpu()


        time = x_mark.cpu().detach().numpy()

        # 确保年、月、日转换为字符串
        years = pd.Series(time[0, :, 0].astype(int)).astype(str)
        months = pd.Series(time[0, :, 1].astype(int)).astype(str).str.zfill(2)  # 使用 pandas.Series
        days = pd.Series(time[0, :, 2].astype(int)).astype(str).str.zfill(2)  # 使用 pandas.Series
        # 使用 str.cat() 连接字符串
        date_strings = years.str.cat(months, sep='-').str.cat(days, sep='-')

        # 转换为时间索引
        date_index = pd.to_datetime(date_strings)

        new_date_range = pd.date_range(start=date_index.min(), end=date_index.max(), freq='D')

        seasonal_full = np.empty((data.shape[0],data.shape[1],len(new_date_range)), dtype=float)  # 存储季节性数据

        # 对每个样本和每个特征进行插值
        for q in range(data.shape[0]):  # 遍历样本
            for j in range(data.shape[1]):  # 遍历样本

                time_series = pd.Series(normalized_tensor[q, j, :], index=date_index)
                time_series = time_series[~time_series.index.duplicated(keep='first')]

                # 对数据进行线性插值
                df_interpolated = time_series.reindex(new_date_range).interpolate(method='polynomial', order=2)

                seasonal_full[q, j, :] = df_interpolated.values
                del df_interpolated,time_series

        seasonal_full=torch.tensor(seasonal_full).float().cuda()

        seasonal_full = seasonal_full.permute(0, 2, 1).contiguous()

        print(seasonal_full.shape)


        infor_out = self.frnet(seasonal_full)
        infor_out=self.mutattn(infor_out)
        out2=infor_out.permute(0,2,1,3)

        out2=out2.reshape(infor_out.shape[0],-1,infor_out.shape[3])
        infor_out=temporal_group_cosine_difference(infor_out,4)


        infor_out=infor_out.view(infor_out.shape[0],infor_out.shape[1],-1)

        infor_out,att2 = self.cat(infor_out)
        infor_out=infor_out.squeeze(2)

        cat = torch.cat((t_out, infor_out), dim=1)

        fianl_out = self.fc1(cat)

        # infor_out = self.fc1(infor_out.squeeze(-1))


        return fianl_out,out1,out2
