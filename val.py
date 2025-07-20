from osgeo import gdal

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import torch.nn as nn
import torch._tensor

import numpy as np

from step_by_step.informer_auto import TSMSPE


def train():
    # train

    # 超参数

    data = torch.tensor(np.load('E:/timenew/BIG/data7_1.npy'))
    date = torch.tensor(np.load('E:/timenew/BIG/datadate7_1.npy'))

    img = gdal.Open('E:/timenew/reallabel/city7_1.tif')

    band_data = []
    for i in range(1, img.RasterCount + 1):  # 获取所有波段
        band1 = img.GetRasterBand(i)  # 获取波段
        data1 = band1.ReadAsArray()  # 读取波段数据
        band_data.append(data1)
    # 将数据转换为 NumPy 数组
    # 如果是单波段图像，则直接提取第一波段
    if len(band_data) == 1:
        array_data = band_data[0]
    else:  # 多波段图像，堆叠为多维数组
        array_data = np.stack(band_data, axis=0)

    # 将 NumPy 数组转换为 PyTorch Tensor
    label = torch.from_numpy(array_data).float()  # 转换为浮点型 Tensor

    Net = TSMSPE.ConvDisRNN(6, 2).cuda()
    Net.load_state_dict(torch.load('md/seed_1TSMSPE.pt'))
    Net.eval()
    m = nn.Softmax(dim=1)


    batch_size = 32  # 每次输入模型的批次大小
    radius = 4  # 假设你裁剪的是 9x9 区域，所以 radius = 4


    # 用于存储拼接的结果

    ylie=[]
    blie = []

    # 创建批次容器
    x_batch = []
    x_mark_batch = []
    labels_batch = []

    for i in range(5, 395):  # 遍历行，跳过边缘
        for j in range(5, 395):  # 遍历列，跳过边缘
            # 获取每个像素的周围 9x9 区域
            patch_data = data[i - radius:i + radius + 1, j - radius:j + radius + 1, :, :]
            patch_date = date[i - radius:i + radius + 1, j - radius:j + radius + 1, :, :]
            patch_label = label[i - radius:i + radius + 1, j - radius:j + radius + 1]


            # 构建一个批次，每次拼接多个小块
            x_batch.append(patch_data.unsqueeze(0).cuda())  # 扩展维度并转到 GPU
            x_mark_batch.append(patch_date.unsqueeze(0)[:, 4, 4, :, :].cuda())  # 获取中心位置的时间标记
            labels_batch.append(patch_label.unsqueeze(0)[:, 4, 4])  # 获取中心位置的标签


            # 每 batch_size 次后输入模型
            if len(x_batch) == batch_size:
                # 拼接成一个大的 batch
                x_batch = torch.cat(x_batch, dim=0)
                x_batch = x_batch.permute(0, 4, 1, 2, 3)

                x_mark_batch = torch.cat(x_mark_batch, dim=0)

                labels_batch = torch.cat(labels_batch, dim=0)


                # 模型预测
                pred = Net(x_batch, x_mark_batch, time_len=x_batch.shape[4])


                y_pred = m(pred)
                y_pred = torch.argmax(y_pred, dim=1).cpu()


                # 将每个预测的结果存储在相应的位置
                for k in range(batch_size):

                    blie.append(labels_batch[k])
                    ylie.append(y_pred[k])

                # 清空批次容器，准备下一批次
                x_batch = []
                x_mark_batch = []
                labels_batch = []

    # 如果还有未处理的部分（不足 batch_size 的部分）
    if len(x_batch) > 0:
        x_batch = torch.cat(x_batch, dim=0)
        x_mark_batch = torch.cat(x_mark_batch, dim=0)
        labels_batch = torch.cat(labels_batch, dim=0)
        x_batch = x_batch.permute(0, 4, 1, 2, 3)


        pred= Net(x_batch, x_mark_batch, time_len=x_batch.shape[4])

        y_pred=m(pred)
        y_pred = torch.argmax(y_pred, dim=1).cpu()

        for k in range(len(x_batch)):

            blie.append(labels_batch[k])
            ylie.append(y_pred[k])

    output_pred = torch.zeros((390, 390), dtype=torch.int64)  # 新的 output_image 大小为 390x390

    output_label = torch.zeros((390, 390), dtype=torch.int64)  # 新的 output_image 大小为 390x390
    index = 0

    # 遍历行和列，从 (5, 395) 开始，边缘忽略
    for i in range(5, 395):
        for j in range(5, 395):

            output_label[i - 5, j - 5] = blie[index]  # 索引需要减去 5，因为新图像从 0 开始
            output_pred[i - 5, j - 5] = ylie[index]  # 索引需要减去 5，因为新图像从 0 开始
            index += 1

    tn, fp, fn, tp = confusion_matrix(output_label.cpu().detach().numpy().flatten(),
                                      output_pred.cpu().detach().numpy().flatten(), labels=[0, 1]).ravel()
    print(tn, fp, fn, tp)
    pre = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    F1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
    print('predpre', pre)
    print('rec', rec)
    print('f1', F1)
    IOU = tp / (tp + fp + 1e-10 + fn)
    print('iou', IOU)

    tn, fp, fn, tp = confusion_matrix(output_label.cpu().detach().numpy().flatten(),
                                      output_pred.cpu().detach().numpy().flatten(), labels=[1, 0]).ravel()
    print(tn, fp, fn, tp)
    pre_0 = tp / (tp + fp + 1e-10)
    rec_0 = tp / (tp + fn + 1e-10)
    F1_0 = 2 * tp / (2 * tp + fp + fn + 1e-10)
    print('predpre', pre_0)
    print('rec', rec_0)
    print('f1', F1_0)
    IOU_0 = tp / (tp + fp + 1e-10 + fn)
    print('iou', IOU_0)
    micof1 = (F1 + F1_0) / 2
    print('micof1', micof1)



if __name__ == '__main__':
    # 在主程序中调用train
    train()