import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from options.train_options import TrainOptions
from models import correlation
from torch.utils.checkpoint import checkpoint

import os
import cv2

opt = TrainOptions().parse()

#應用偏移量(offset)，生成標準化座標
def apply_offset(offset):   # offset: (batch_size, channel, height, width)
    sizes = list(offset.size()[2:])
    # 使用meshgrid創建座標，每個點都會對應到一個空間中的座標
    grid_list = torch.meshgrid(
        [torch.arange(size, device=offset.device) for size in sizes], indexing='ij')
    
    grid_list = reversed(grid_list) #反轉座標順序
    # 應用偏移量，將座標轉換成float，並且在第一維度添加新的維度，再加上偏移量
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...] for dim, grid in enumerate(grid_list)]
    # 標準化座標，使座標在[-1, 1]之間
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0 for grid, size in zip(grid_list, reversed(sizes))]
    
    # 將標準化座標沿著最後一個維度堆疊起來，形成新的張量
    return torch.stack(grid_list, dim=-1)

# Total Variation Loss(全變差損失)，減少雜訊
def TVLoss(x):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))

# TVLoss進階版，考慮了遮罩，使得此平滑損失只在遮罩範圍內生效
def TVLoss_v2(x, mask):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    h, w = mask.size(2), mask.size(3)

    tv_h = tv_h * mask[:, :, :h-1, :]
    tv_w = tv_w * mask[:, :, :, :w-1]

    if torch.sum(mask) > 0:
        return (torch.sum(torch.abs(tv_h)) + torch.sum(torch.abs(tv_w))) / torch.sum(mask)
    else:
        return torch.sum(torch.abs(tv_h)) + torch.sum(torch.abs(tv_w))
    

# 針對光流場的損失
def SquareTVLoss(flow):
    flow_x, flow_y = torch.split(flow, 1, dim=1)    # 可以對x, y方向進行單獨的處理
    # 計算差
    flow_x_diff_left = flow_x[:, :, :, 1:] - flow_x[:, :, :, :-1]
    flow_x_diff_right = flow_x[:, :, :, :-1] - flow_x[:, :, :, 1:]
    # 裁剪邊界，讓x與y符合大小
    flow_x_diff_left = flow_x_diff_left[...,1:-1,:-1]
    flow_x_diff_right = flow_x_diff_right[...,1:-1,1:]

    # 計算差
    flow_y_diff_top = flow_y[:, :, 1:, :] - flow_y[:, :, :-1, :]
    flow_y_diff_bottom = flow_y[:, :, :-1, :] - flow_y[:, :, 1:, :]
    # 裁剪邊界，讓x與y符合大小
    flow_y_diff_top = flow_y_diff_top[...,:-1,1:-1]
    flow_y_diff_bottom = flow_y_diff_bottom[...,1:,1:-1]

    # 各方向的差異
    left_top_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_top))
    left_bottom_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_bottom))
    right_top_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_top))
    right_bottom_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_bottom))

    return torch.mean(left_top_diff+left_bottom_diff+right_top_diff+right_bottom_diff)

# SquareTVLoss進階版，利用interval控制計算差的範圍，使損失在不同尺度下也能有效控制流的平滑效果
def SquareTVLoss_v2(flow, interval_list=[1,5]):
    flow_x, flow_y = torch.split(flow, 1, dim=1)

    tvloss = 0
    for interval in interval_list:
        flow_x_diff_left = flow_x[:, :, :, interval:] - flow_x[:, :, :, :-interval]
        flow_x_diff_right = flow_x[:, :, :, :-interval] - flow_x[:, :, :, interval:]
        flow_x_diff_left = flow_x_diff_left[...,interval:-interval,:-interval]
        flow_x_diff_right = flow_x_diff_right[...,interval:-interval,interval:]

        flow_y_diff_top = flow_y[:, :, interval:, :] - flow_y[:, :, :-interval, :]
        flow_y_diff_bottom = flow_y[:, :, :-interval, :] - flow_y[:, :, interval:, :]
        flow_y_diff_top = flow_y_diff_top[...,:-interval,interval:-interval]
        flow_y_diff_bottom = flow_y_diff_bottom[...,interval:,interval:-interval]

        left_top_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_top))
        left_bottom_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_bottom))
        right_top_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_top))
        right_bottom_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_bottom))

        tvloss += torch.mean(left_top_diff+left_bottom_diff+right_top_diff+right_bottom_diff)

    return tvloss

'''------------------------骨幹(backbone)----------------------------------'''
#殘差網路骨幹
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        # layer容器，將多層layer組合成神經網路
        self.block = nn.Sequential(
            # 應用於風格轉換、生成對抗網路的正規化。減少圖像中每個通道的平均顏色和對比度的差異
            # 不使用Batch Normalization是因為BN不適合小的batch size，且更適合用於分類與回歸等卷積任務。
            nn.InstanceNorm2d(in_channels), 
            nn.ReLU(inplace=True),  # 將<0的值變0，inplace=true使其不會創建新的張量來儲存結果。
            # 其他激活函數Sigmoid, Tanh在輸入值很大或很小時會導致梯度消失的問題，因此使用ReLU
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),  #卷積層，(輸入通道數, 輸出通道數, 卷積核大小, 補充像素32 => 32, 不會有bias)

            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        # 輸入形狀會是 (batch_size, in_channels, height, width)
        # return checkpoint(self.block, x, use_reentrant=False) + x    #特徵 + 原本輸出 = 殘差原理
        return self.block(x) + x    #特徵 + 原本輸出 = 殘差原理

    

# 下採樣。減少特徵圖的空間維度，同時增加特徵通道的數量。
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        # layer容器，將多層layer組合成神經網路
        self.block = nn.Sequential(     
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)    # stride用來控制卷積核的移動步長，因此經過這裡高寬會減半
        )

    def forward(self, x):
        # 輸入形狀會是 (batch_size, in_channels, height, width)
        return self.block(x)

# 取得特徵
class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64, 128, 256, 256, 256]): # 定義每層的輸出通道數
        super(FeatureEncoder, self).__init__()
        self.encoders = []  # 儲存編碼器

        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:   # 使用前一層輸出通道數當作輸入通道數
                encoder = nn.Sequential(DownSample(chns[i-1], out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)    #自動參數追蹤，並且用法跟List一樣

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features
    

# 建立特徵金字塔(FPN)，有效地融合多層次的特徵以捕捉不同尺度的信息
# 根據FeatureEncoder的輸出結果融合特徵，因此輸入會是chns通道數
class RefinePyramid(nn.Module):
    def __init__(self, chns=[64, 128, 256, 256, 256], fpn_dim=256): # 這裡的chns是輸入的通道數
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # Adaptive Layers（自適應層）
        self.adaptive = []

        for in_chns in list(reversed(chns)):    # 從最高層開始，使高層的語意特徵能融合到低層中
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)

        self.adaptive = nn.ModuleList(self.adaptive)

        # 平滑層，平滑融合後的特徵
        self.smooth = []
        for i in range(len(chns)):
            smooth_layer = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x

        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)    # 按照順序調用自適應層
            
            if last_feature is not None:    # 根據尺寸進行上採樣(upsampling)，將圖片放大兩倍
                feature = feature + \
                    F.interpolate(last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)   # 按照順序調用平滑層
            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))    # 反轉回原來的排列

# 對人體不同部分進行特徵提取、細化、融合
class AFlowNet_Vitonhd_lrarms(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256):   # num_pyramid = 5
        super(AFlowNet_Vitonhd_lrarms, self).__init__()
        self.netLeftMain = []   # 左臂主要網路
        self.netTorsoMain = []  # 軀幹主要網路
        self.netRightMain = []  # 右臂主要網路

        self.netLeftRefine = []     # 左臂細化處理
        self.netTorsoRefine = []    # 軀幹細化處理
        self.netRightRefine = []    # 右臂細化處理

        self.netAttentionRefine = []    # 處理注意力機制的細化層
        self.netPartFusion = []         # 將不同部件融合在一起的層
        self.netSeg = []                # 生成分割圖像層，對各部件圖像進行切割

        '''LeakyReLU中的negative_slope=0.1 使ReLU不會有死神經元的出現, 當輸出小於0時會乘以0.1, 是針對ReLU的變體'''
        for i in range(num_pyramid):
            # 左臂的主要網路處理層，49個通道輸入，經過多次卷積層、激活函數使輸出變成2個通道的特徵圖
            netLeftMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), 
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            # 軀幹的主要網路處理層，49個通道輸入，經過多次卷積層、激活函數使輸出變成2個通道的特徵圖
            netTorsoMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            # 右臂的主要網路處理層，49個通道輸入，經過多次卷積層、激活函數使輸出變成2個通道的特徵圖
            netRightMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            #左臂細化處理，將輸入特徵逐步縮小並提取出有用的特徵
            netRefine_left_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            #軀幹細化處理，將輸入特徵逐步縮小並提取出有用的特徵
            netRefine_torso_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            #右臂細化處理，將輸入特徵逐步縮小並提取出有用的特徵
            netRefine_right_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            #對多個部位的特徵進行細化處理，基於已經合併的部件特徵圖
            netAttentionRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=4 * fpn_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh() #限縮範圍，自動將輸出對應到[-1, 1]
            )

            # 切割圖像任務
            netSeg_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=fpn_dim*2, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=7, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh() #限縮範圍，自動將輸出對應到[-1, 1]
            )

            # 利用殘差網路對特徵進行融合處理
            partFusion_layer = torch.nn.Sequential(
                nn.Conv2d(fpn_dim*3, fpn_dim, kernel_size=1),
                ResBlock(fpn_dim)
            )

            self.netLeftMain.append(netLeftMain_layer)
            self.netTorsoMain.append(netTorsoMain_layer)
            self.netRightMain.append(netRightMain_layer)

            self.netLeftRefine.append(netRefine_left_layer)
            self.netTorsoRefine.append(netRefine_torso_layer)
            self.netRightRefine.append(netRefine_right_layer)

            self.netAttentionRefine.append(netAttentionRefine_layer)
            self.netPartFusion.append(partFusion_layer)
            self.netSeg.append(netSeg_layer)

        self.netLeftMain = nn.ModuleList(self.netLeftMain)
        self.netTorsoMain = nn.ModuleList(self.netTorsoMain)
        self.netRightMain = nn.ModuleList(self.netRightMain)

        self.netLeftRefine = nn.ModuleList(self.netLeftRefine)
        self.netTorsoRefine = nn.ModuleList(self.netTorsoRefine)
        self.netRightRefine = nn.ModuleList(self.netRightRefine)

        self.netAttentionRefine = nn.ModuleList(self.netAttentionRefine)
        self.netPartFusion = nn.ModuleList(self.netPartFusion)
        self.netSeg = nn.ModuleList(self.netSeg)
        self.softmax = torch.nn.Softmax(dim=1)  # 輸入的張量轉換為概率分佈，dim=1是對每一個樣本進行softmax

    # 預測
    def forward(self, x, x_edge, x_full, x_edge_full, x_warps, x_conds, preserve_mask, warp_feature=True):
        last_flow = None        # 上次的flow訊息
        last_flow_all = []      # 過往 flow 訊息
        delta_list = []         # flow 變化量
        x_all = []              # warping 後的所有輸入圖像。通常是從原始圖像經過 flow 變換後得到的結果
        x_edge_all = []         # 儲存經過 warping 後的邊緣檢測結果。用來識別物體的邊界
        x_full_all = []         # 所有完整圖像的處理結果，包含未做過 warping 或做過其他變化的處理結果
        x_edge_full_all = []    # 儲存完整圖像的邊緣檢測結果，針對的是完整圖像的邊緣
        attention_all = []      # 儲存每一個步驟中計算出的注意力權重，並強調關鍵特徵。
        seg_list = []           # 儲存每個步驟中的分割結果，涉及到對不同區域進行分類或標記
        delta_x_all = []        # x 方向的 flow 變化
        delta_y_all = []        # y 方向的 flow 變化

        # 邊緣檢測濾波器
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]

        #將濾波器儲存在 weight_array 中
        weight_array = np.ones([3, 3, 1, 4])    #建立初始矩陣，形狀為(3, 3, 1, 4)，(卷積核高度, 卷積核寬度, 通道數, 表示有 4 個不同的濾波器)
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        # 轉換為 torch 浮點張量，在GPU上運行，並改變其形狀為(4, 1, 3, 3)
        # weight_array = torch.cuda.FloatTensor(weight_array).permute(3, 2, 0, 1)
        weight_array = torch.tensor(weight_array, dtype=torch.float32, device='cuda').permute(3, 2, 0, 1)
        # 設置可訓練參數，requires_grad=false，代表此權重矩陣不會被更新，也就是濾波器是固定的
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

        for i in range(len(x_warps)):
            # 反向選取元素
            '''
            torch.Size([16, 256, 8, 6])], 
            torch.Size([16, 256, 16, 12]),
            torch.Size([16, 256, 32, 24]), 
            torch.Size([16, 256, 64, 48]), 
            torch.Size([16, 256, 128, 96])
            '''
            x_warp = x_warps[len(x_warps) - 1 - i]  # 4-i   [4, 3, 2, 1, 0] size: 16
            # print("x_warp: ", x_warp[0].shape)
            '''
            torch.Size([16, 256, 16, 12]),
            torch.Size([16, 256, 32, 24]), 
            torch.Size([16, 256, 64, 48]), 
            torch.Size([16, 256, 128, 96]),
            torch.Size([16, 256, 256, 192])
            '''
            x_cond = x_conds[len(x_warps) - 1 - i]  # 4-i   [4, 3, 2, 1, 0] size: 16
            # print("x_cond: ", x_cond[0].shape)
            # 將 x_cond 和 x_warp 在第0個維度上重複三次
            x_warp_concate = torch.cat([x_warp,x_warp,x_warp],0)    # size: 48，[48, 256, height, width]
            x_cond_concate = torch.cat([x_cond,x_cond,x_cond],0)    # size: 48，[48, 256, height, width]

            # 前一層得到的 flow 提供了每個像素在空間的位移，因此重新採樣是根據 last_flow 對 x_warp_concate 進行位移，實現對輸入的動態變形
            if last_flow is not None and warp_feature:
                # 對輸入張量進行重新採樣，其目的是利用 last_flow 來改變 x_warp_concate 的空間位置。
                # detach()確保不會影響梯度計算。bilinear: 雙線性插值。border: 在邊界外取樣時使用邊界像素的值來進行填充
                # [batch size, height, width, channels]
                x_warp_after = F.grid_sample(x_warp_concate, last_flow.detach().permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
            else:
                x_warp_after = x_warp_concate

            # 計算 x_warp_after 和 x_cond_concate 張量之間的相關性。指定為 leaky_relu 使其減少神經元死亡問題
            tenCorrelation = F.leaky_relu(input=correlation.FunctionCorrelation(
                tenFirst=x_warp_after, tenSecond=x_cond_concate, intStride=1), negative_slope=0.1, inplace=False)
            # for i = 0, tenCorrelation.shape: [48, 49, 8, 6]
            # print("tenCorrelation: ", tenCorrelation.shape)
            
            # 取得張量 x_cond 的第一維大小(batch size)，用於分割 tenCorrelation
            bz = x_cond.size(0)     

            #分割成左、軀幹、右
            left_tenCorrelation = tenCorrelation[0:bz]
            torso_tenCorrelation = tenCorrelation[bz:2*bz]
            right_tenCorrelation = tenCorrelation[2*bz:]

            # 利用主要網路進行計算第i層的flow，將來自不同區域的相關性特徵轉換為流
            # left_flow = self.netLeftMain[i](left_tenCorrelation)
            # torso_flow = self.netTorsoMain[i](torso_tenCorrelation)
            # right_flow = self.netRightMain[i](right_tenCorrelation)
            left_flow = checkpoint(self.netLeftMain[i], left_tenCorrelation, use_reentrant=False)
            torso_flow = checkpoint(self.netTorsoMain[i], torso_tenCorrelation, use_reentrant=False)
            right_flow = checkpoint(self.netRightMain[i], right_tenCorrelation, use_reentrant=False)

            # 依第一維度進行拼接
            flow = torch.cat([left_flow,torso_flow,right_flow],0)

            # 儲存流變化
            delta_list.append(flow)
            # 用於修正或調整流的值
            flow = apply_offset(flow)

            # F.grid_sample 使用 last_flow 對當前 flow 進行插值
            if last_flow is not None:
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border', align_corners=False)
            else:
                flow = flow.permute(0, 3, 1, 2) # 維度變換

            # 更新last_flow
            last_flow = flow

            # 使用當前 flow 對 x_warp_concate 進行變形、插值
            # [batch size, height, width, channels]
            x_warp_concate = F.grid_sample(x_warp_concate, flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
            # 合併各部件張量
            left_concat = torch.cat([x_warp_concate[0:bz], x_cond_concate[0:bz]], 1)
            torso_concat = torch.cat([x_warp_concate[bz:2*bz], x_cond_concate[bz:2*bz]],1)
            right_concat = torch.cat([x_warp_concate[2*bz:], x_cond_concate[2*bz:]],1)

            '''----------------------------------------------------------------------------------------------------------------------------------------'''

            # x_warp_concate、x_cond合併注意力特徵
            x_attention = torch.cat([x_warp_concate[0:bz],x_warp_concate[bz:2*bz],x_warp_concate[2*bz:],x_cond],1)
            # 將融合的特徵進行注意力計算
            fused_attention = self.netAttentionRefine[i](x_attention)
            # fused_attention = checkpoint(self.netAttentionRefine[i], x_attention, use_reentrant=False)
            # 限制於[-1, 1]
            fused_attention = self.softmax(fused_attention)
            # fused_attention = checkpoint(self.softmax, fused_attention, use_reentrant=False)

            # 進行細化流計算
            # left_flow = self.netLeftRefine[i](left_concat)
            # torso_flow = self.netTorsoRefine[i](torso_concat)
            # right_flow = self.netRightRefine[i](right_concat)  
            left_flow = checkpoint(self.netLeftRefine[i], left_concat, use_reentrant=False)
            torso_flow = checkpoint(self.netTorsoRefine[i], torso_concat, use_reentrant=False)
            right_flow = checkpoint(self.netRightRefine[i], right_concat, use_reentrant=False)   

            # 合併 flow
            flow = torch.cat([left_flow,torso_flow,right_flow],0)
            # 儲存流變化
            delta_list.append(flow)
            # 用於修正或調整流的值
            flow = apply_offset(flow)
            # F.grid_sample 使用 last_flow 對當前 flow 進行插值
            flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border', align_corners=False)
            
            '''----------------------------------------------------------------------------------------------------------------------------------------'''

            # 根據之前計算的 fused_attention 進行加權合併，將每個區域的流乘以相應的注意力，最終形成一個融合的流。
            fused_flow = flow[0:bz] * fused_attention[:,0:1,...] + \
                flow[bz:2*bz] * fused_attention[:,1:2,...] + \
                flow[2*bz:] * fused_attention[:,2:3,...]
            # 對融合過後的 flow, attention 進行上採樣
            last_fused_flow = F.interpolate(fused_flow, scale_factor=2, mode='bilinear')
            fused_attention = F.interpolate(fused_attention, scale_factor=2, mode='bilinear')
            # 紀錄注意力權重
            attention_all.append(fused_attention)

            # 上採樣，根據當前迭代的層數 i 和總的 x_warps 數量，計算縮放程度，用於調整圖像的尺寸
            cur_x_full = F.interpolate(x_full, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            # 進行變形操作
            cur_x_full_warp = F.grid_sample(cur_x_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
            # 將變形過後的圖像添加到 x_full_all
            x_full_all.append(cur_x_full_warp)

            #對遮罩圖像進行插值與變形
            cur_x_edge_full = F.interpolate(x_edge_full, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_edge_full_warp = F.grid_sample(cur_x_edge_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
            x_edge_full_all.append(cur_x_edge_full_warp)

            # 對當前的 flow 進行上採樣以便用於下一次迭代
            last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
            last_flow_all.append(last_flow)

            # 對原始圖像進行插值與變形 [batch size, height, width, channel]
            # print("x: ", x.shape)
            cur_x = F.interpolate(x, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_warp = F.grid_sample(cur_x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
            # print("cur_x_warp: ", cur_x_warp.shape)
            x_all.append(cur_x_warp)

            # 對原始遮罩圖像進行插值與變形
            cur_x_edge = F.interpolate(x_edge, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_warp_edge = F.grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
            x_edge_all.append(cur_x_warp_edge)

            # 按通道拆分為 x 和 y 分量
            flow_x, flow_y = torch.split(last_flow, 1, dim=1)
            # 分別對 x, y 進行卷積操作，計算方向的變化
            delta_x = F.conv2d(flow_x, self.weight)
            delta_y = F.conv2d(flow_y, self.weight)
            # 保存計算結果
            delta_x_all.append(delta_x)
            delta_y_all.append(delta_y)

            '''----------------------------------------------------------------------------------------------------------------------------------------'''

            # 處理分割預測
            # 將 preserve_mask 上採樣到當前大小，並且它是一個遮罩，用於指示哪些部分應該被保留或忽略
            cur_preserve_mask = F.interpolate(preserve_mask, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            # 獲取的圖像特徵
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            # 擴展通道並上採樣
            x_warp = torch.cat([x_warp,x_warp,x_warp],0)
            x_warp = F.interpolate(x_warp, scale_factor=2, mode='bilinear')
            x_cond = F.interpolate(x_cond, scale_factor=2, mode='bilinear')

            # 根據 last_flow 對 x_warp 進行變形，
            x_warp = F.grid_sample(x_warp, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border', align_corners=False)
            
            # 分割區域成左、軀幹、右
            x_warp_left = x_warp[0:bz]
            x_warp_torso = x_warp[bz:2*bz]
            x_warp_right = x_warp[2*bz:]

            # 分割遮罩成左、軀幹、右
            x_edge_left = cur_x_warp_edge[0:bz]
            x_edge_torso = cur_x_warp_edge[bz:2*bz]
            x_edge_right = cur_x_warp_edge[2*bz:]
            
            # x_edge_left = x_edge_left.expand(-1, 256, -1, -1) # ??????
            # print("x_warp_left: ", x_warp_left.shape)
            # print("x_edge_left: ", x_edge_left.shape)
            # 邊緣信息和 cur_preserve_mask 對每一部分的 x_warp 進行加權，強調邊界並抑制不需要的區域
            x_warp_left = x_warp_left * x_edge_left * (1-cur_preserve_mask)
            x_warp_torso = x_warp_torso * x_edge_torso * (1-cur_preserve_mask)
            x_warp_right = x_warp_right * x_edge_right * (1-cur_preserve_mask)

            # 合併區域
            x_warp = torch.cat([x_warp_left,x_warp_torso,x_warp_right],1)
            # for j in range(2):
            #     print(x_warp[j].shape)
            # 特徵融合
            # x_warp = self.netPartFusion[i](x_warp)
            x_warp = checkpoint(self.netPartFusion[i], x_warp, use_reentrant=False)
                # seg_save = np.sum(seg[j].cpu().numpy(), axis=0)
                # seg_save = (((seg_save+1)/2) * 255).astype(np.uint8)
                # seg_save = np.transpose(seg_save, (1, 2, 0))

                # os.makedirs('sample/test_warping/seg/item/', exist_ok = True)
                # save_path = f'sample/test_warping/seg/item/{i}___{j}.png'
            #     cv2.imwrite(save_path, seg_save)

            # 特徵拼接，在第二個維度拼接(Channel)
            concate = torch.cat([x_warp,x_cond],1)
            # 傳遞到分割網絡 netSeg 中進行分割預測
            # seg = self.netSeg[i](concate)
            seg = checkpoint(self.netSeg[i], concate, use_reentrant=False)
            # for j in range(2):
            #     print(seg[j].shape)
            #     seg_save = np.sum(seg[j].cpu().numpy(), axis=0)
            #     seg_save = (((seg_save+1)/2) * 255).astype(np.uint8)
            #     # seg_save = np.transpose(seg_save, (1, 2, 0))

            #     os.makedirs('sample/test_warping/seg/item/', exist_ok = True)
            #     save_path = f'sample/test_warping/seg/item/{i}___{j}.png'
            #     cv2.imwrite(save_path, seg_save)
            # 保存
            seg_list.append(seg)

        '''
        last_flow:  torch.Size([12, 2, 256, 192])
        last_flow_all:  torch.Size([12, 2, 16, 12])
        delta_list:  torch.Size([12, 2, 8, 6])
        x_all:  torch.Size([12, 3, 16, 12])
        x_edge_all:  torch.Size([12, 1, 16, 12])
        delta_x_all:  torch.Size([12, 4, 14, 10])
        delta_y_all:  torch.Size([12, 4, 14, 10])
        x_full_all:  torch.Size([4, 3, 16, 12])
        x_edge_full_all:  torch.Size([4, 1, 16, 12])
        attention_all:  torch.Size([4, 3, 16, 12])
        seg_list:  torch.Size([4, 7, 16, 12])
        '''

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, x_full_all, x_edge_full_all, attention_all, seg_list

# 用於處理圖像變換任務，結合特徵提取、金字塔網絡和 flow net 等多種技術，以達成高效的圖像處理和生成
class AFWM_Vitonhd_lrarms(nn.Module):
    def __init__(self, opt, input_nc, clothes_input_nc=3):
        super(AFWM_Vitonhd_lrarms, self).__init__()
        # 設置濾波器數量
        num_filters = [64, 128, 256, 256, 256]
        # 設置金字塔維度
        fpn_dim = 256

        # 特徵提取模型建置
        self.image_features = FeatureEncoder(clothes_input_nc+1, num_filters)   #(4, [64, 128, 256, 256, 256])
        self.cond_features = FeatureEncoder(input_nc, num_filters)              #(51,[64, 128, 256, 256, 256])
        # print("input_nc: ", input_nc)
        
        # 金字塔網路建置
        self.image_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        self.cond_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        
        # flow網路建置
        self.aflow_net = AFlowNet_Vitonhd_lrarms(len(num_filters))

        # 儲存初始學習率
        self.old_lr = opt.lr
        # 縮小學習率
        self.old_lr_warp = opt.lr*0.2

    def forward(self, cond_input, image_input, image_edge, image_label_input, image_input_left, image_input_torso, \
                image_input_right, image_edge_left, image_edge_torso, image_edge_right, preserve_mask):
        # 將特徵結合
        image_input_concat = torch.cat([image_input, image_label_input],1)  # [16, 3, 256, 192] + [16, 1, 256, 192] = [16, 4, 256, 192]
        # cond_input = F.interpolate(cond_input, size=image_input_concat.shape[2:], mode='bilinear', align_corners=False)

        # 提取特徵
        # print("image_input_concat", image_input_concat.shape)
        # image_pyramids = self.image_FPN(self.image_features(image_input_concat))
        image_pyramids = checkpoint(self.image_FPN, checkpoint(self.image_features, image_input_concat, use_reentrant=False), use_reentrant=False)
        # import test
        # print("image_pyramids", test.get_tuple_shapes(image_pyramids))
        '''
        len(self.image_features([16, 4, 256, 192])) = 5 層特徵      
        len(image_pyramids) = 5 層特徵
        image_pyramids.shape[
            torch.Size([16, 256, 128, 96]), 
            torch.Size([16, 256, 64, 48]), 
            torch.Size([16, 256, 32, 24]), 
            torch.Size([16, 256, 16, 12]), 
            torch.Size([16, 256, 8, 6])]
            [batch size, channels, height, width]
        '''
        
        # cond_pyramids = self.cond_FPN(self.cond_features(cond_input))  # cond_input: [16, 51, 512, 384]
        cond_pyramids = checkpoint(self.cond_FPN, checkpoint(self.cond_features, cond_input, use_reentrant=False), use_reentrant=False)  # cond_input: [16, 51, 512, 384]
        '''
        len(self.image_features([16, 51, 512, 384])) = 5 層特徵      
        len(image_pyramids) = 5 層特徵
        cond_pyramids.shape[
            torch.Size([16, 256, 256, 192]), 
            torch.Size([16, 256, 128, 96]), 
            torch.Size([16, 256, 64, 48]), 
            torch.Size([16, 256, 32, 24]), 
            torch.Size([16, 256, 16, 12])]
            [batch size, channels, height, width]
        '''

        # 結合左臂、軀幹、右臂區域
        image_concat = torch.cat([image_input_left,image_input_torso,image_input_right],0)
        '''
        [batch size, channels, height, width]
        image_concat.shape: [48, 3, 256, 192]
        使用batch size拼接將來自不同區域的圖片組合在一起。這樣的拼接方式可以一次處理不同區域的所有圖像。
        '''
        # 結合左臂、軀幹、右臂遮罩
        image_edge_concat = torch.cat([image_edge_left, image_edge_torso, image_edge_right],0)
        '''
        [batch size, channels, height, width]
        image_edge_concat.shape: [48, 1, 256, 192]
        使用batch size拼接將來自不同區域的圖片組合在一起。這樣的拼接方式可以一次處理不同區域的所有圖像。
        '''

        # 進行flow net
        last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
            x_full_all, x_edge_full_all, attention_all, seg_list = self.aflow_net(image_concat, \
            image_edge_concat, image_input, image_edge, image_pyramids, cond_pyramids, \
            preserve_mask)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
                x_full_all, x_edge_full_all, attention_all, seg_list

    # 更新主要學習率
    def update_learning_rate(self, optimizer):
        # 計算每次學習率要減少的量
        lrd = opt.lr / opt.niter_decay
        # 降低學習率
        # lr = self.old_lr - lrd
        lr = max(self.old_lr - lrd, 1e-6)
        # 更新優化器內的 lr 參數
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        #輸出學習率更新的詳細信息，追蹤學習率的變化
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        # 更新 lr
        self.old_lr = lr

    # 更新服裝變形的學習率
    def update_learning_rate_warp(self, optimizer):
        # 減少量是原學習率的20%，用於更新與圖像變形相關的學習率，使學習更加穩定
        lrd = 0.2 * opt.lr / opt.niter_decay
        # lr = self.old_lr_warp - lrd
        lr = max(self.old_lr_warp - lrd, 1e-6)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
        self.old_lr_warp = lr
