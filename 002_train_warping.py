import os
import torch
import functools
import torch.nn as nn
import time
import torch.nn.functional as F
import cv2
import datetime
import numpy as np
from tqdm import tqdm
from options.train_options import TrainOptions
from data import aligned_dataset
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from models import AFWM
from models.networks import SpectralDiscriminator, load_checkpoint_parallel, VGGLoss, GANLoss, set_requires_grad, save_checkpoint


def CreateDataset(opt):
    dataset = aligned_dataset.AlignedDataset()
    dataset.initialize(opt)
    
    return dataset

opt = TrainOptions().parse()

run_path = 'runs/'+opt.name         # 'runs/flow'
sample_path = 'sample/'+opt.name    # 'sample/flow'

# 創建資料夾, exist_ok=True: 若沒有資料夾不會報錯
os.makedirs(sample_path, exist_ok = True)
os.makedirs(run_path, exist_ok = True)

# model 會放在這
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')     # './checkpoints'

# 設定GPU編號
torch.cuda.set_device(opt.local_rank)

# 用GPU跑
device = torch.device(f'cuda:{opt.local_rank}')

train_data = CreateDataset(opt)     # 11632 不是 11647是因為batchsize分割時data沒辦法被整除
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataset_size = len(train_loader)    # 727
# print("train_data len:", len(train_data))

warp_model = AFWM.AFWM_Vitonhd_lrarms(opt, 51)
warp_model.train()
warp_model.cuda()

# 將變形模組的 Batch Normalization 轉成 SyncBatchNorm，用於多張GPU時。
if opt.PBAFN_warp_checkpoint is not None:
    load_checkpoint_parallel(warp_model, opt.PBAFN_warp_checkpoint)
# warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

# 在多GPU上進行 DistributedDataParallel 包裝，由於只有一張GPU，因此不需要，直接指定就好
# if opt.isTrain and len(opt.gpu_ids):
#     model = torch.nn.parallel.DistributedDataParallel(warp_model, device_ids=[opt.local_rank])

# 將模型移動到GPU上
model = warp_model.to(device)

'''與主模型有關'''
# 提取可訓練參數
params_warp = [p for p in model.parameters()]   # 遍歷所有參數
optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999)) # 設置Adam優化器，設置betas控制一階、二階動量估計平滑程度

# 判別器模型，做譜歸一化，控制訓練時的梯度，讓模型更穩定
discriminator = SpectralDiscriminator(opt, input_nc=59, ndf=64, n_layers=3, # 輸入影像層數、卷積層的初始通道數、鑑別器的卷積層數
                                    # 正規化層使用 InstanceNorm2d，affine 允許使用學習的縮放和平移參數，track_running_stats 允許追蹤均值和變異數
                                    norm_layer=functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True), 
                                    use_sigmoid=False)  # 不允許使用Sigmoid激活函數，在對抗網路中不需要將輸出限制在[0, 1]

discriminator.train()
discriminator.cuda()

# 讓模型在訓練或推理時使用之前訓練過的參數
if opt.pretrain_checkpoint_D is not None:
    load_checkpoint_parallel(discriminator, opt.pretrain_checkpoint_D)

# 在使用多張GPU時需要同步批量歸一化、分佈式數據並行
# discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator).to(device)
# if opt.isTrain and len(opt.gpu_ids):
#     discriminator = torch.nn.parallel.DistributedDataParallel(
#         discriminator, device_ids=[opt.local_rank])

# 只有一張GPU只需要將模型移至GPU上進行訓練就好了
discriminator = discriminator.to(device)

'''與判別器有關'''
# 定義並初始化優化器，用來訓練鑑別器，lambda 條件用來過濾那些標記為需要梯度計算的參數，parameters 只會返回可訓練參數
params_D = list(filter(lambda p: p.requires_grad, discriminator.parameters()))

# 初始化 Adam 優化器
optimizer_D = torch.optim.Adam(params_D, lr=opt.lr_D, betas=(opt.beta1, 0.999))

criterionL1 = nn.L1Loss()               # 平均誤差損失(MAE)
criterionVGG = VGGLoss()
criterionLSGANloss = GANLoss().cuda()
softmax = torch.nn.Softmax(dim=1)       # dim表示對每一行計算softmax，使所有元素都在[0, 1]之間

# 記錄訓練過程中的數據（如損失、學習率等），並將這些數據保存到指定位置
writer = SummaryWriter(run_path)
# print('#training images = %d' % dataset_size)


start_epoch = 1 # 起始epoch
epoch_iter = 0  # 訓練了幾次epoch
total_steps = (start_epoch-1) * dataset_size + epoch_iter   # 總步數
step = 0    # 初始步數
step_per_batch = dataset_size   # epoch 中的步數（batch size）

# for epoch in trange(start_epoch, opt.niter + opt.niter_decay + 1, desc="Epoch Progress"):
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    result_out = True
    # for epoch in range(start_epoch, 11):    # 訓練10次
    epoch_start_time = time.time()  # 紀錄每次epoch開始時間
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    # train_data.set_epoch(epoch)   # 分布式訓練所需
    # for i, data in enumerate(train_loader):
    for i, data in enumerate(tqdm(train_loader, desc="Training", total=len(train_loader))):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1

        pre_clothes_edge = data['edge']                         # 服裝遮罩
        clothes = data['cloth']                                 # 原始服裝影像
        clothes = clothes * pre_clothes_edge                    # 只保留遮罩區域的服裝特徵
        person_clothes_edge = data['person_clothes_mask']       # 人物穿著服裝的邊緣
        real_image = data['image']                              # 人物的原始影像
        person_clothes = real_image * person_clothes_edge       # 提取人物穿著的服裝區域
        pose = data['pose']                                     # 人物姿勢數據

        # dencepose 預處理
        size = data['cloth'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])  # DensePose的One-Hot尺寸
        densepose = torch.zeros(oneHot_size1, dtype=torch.float, device='cuda')
        data['densepose'] = (data['densepose'] / 229.0 * 24.0).round().long()
        densepose = densepose.scatter_(1, data['densepose'].data.long().cuda(), 1.0)
        
        densepose = densepose * 2.0 - 1.0       # 將值縮放至 [-1, 1]
        densepose_fore = data['densepose']/24.0 # DensePose前景

        # 服裝分割與遮罩
        left_cloth_sleeve_mask = data['flat_clothes_left_mask']
        cloth_torso_mask = data['flat_clothes_middle_mask']
        right_cloth_sleeve_mask = data['flat_clothes_right_mask']

        # 生成遮罩
        part_mask = torch.cat([left_cloth_sleeve_mask,cloth_torso_mask,right_cloth_sleeve_mask],0)
        part_mask = (torch.sum(part_mask,dim=(2,3),keepdim=True)>0).float().cuda()

        clothes_left = clothes * left_cloth_sleeve_mask
        clothes_torso = clothes * cloth_torso_mask
        clothes_right = clothes * right_cloth_sleeve_mask

        cloth_parse_for_d = data['flat_clothes_label'].cuda()
        cloth_parse_vis = torch.cat([cloth_parse_for_d,cloth_parse_for_d,cloth_parse_for_d],1)

        person_clothes_left_sleeve_mask = data['person_clothes_left_mask']
        person_clothes_torso_mask = data['person_clothes_middle_mask']
        person_clothes_right_sleeve_mask = data['person_clothes_right_mask']
        person_clothes_mask_concate = torch.cat([person_clothes_left_sleeve_mask,person_clothes_torso_mask,person_clothes_right_sleeve_mask],0)

        seg_label_tensor = data['seg_gt']
        seg_gt_tensor = (seg_label_tensor / 6 * 2 -1).float()
        seg_label_onehot_tensor = data['seg_gt_onehot'] * 2 - 1.0

        seg_label_tensor = seg_label_tensor.cuda()
        seg_gt_tensor = seg_gt_tensor.cuda()
        seg_label_onehot_tensor = seg_label_onehot_tensor.cuda()

        person_clothes = person_clothes.cuda()
        person_clothes_edge = person_clothes_edge.cuda()
        pose = pose.cuda()

        clothes = clothes.cuda()
        clothes_left = clothes_left.cuda()
        clothes_torso = clothes_torso.cuda()
        clothes_right = clothes_right.cuda()

        pre_clothes_edge = pre_clothes_edge.cuda()
        left_cloth_sleeve_mask = left_cloth_sleeve_mask.cuda()
        cloth_torso_mask = cloth_torso_mask.cuda()
        right_cloth_sleeve_mask = right_cloth_sleeve_mask.cuda()

        person_clothes_left_sleeve_mask = person_clothes_left_sleeve_mask.cuda()
        person_clothes_torso_mask = person_clothes_torso_mask.cuda()
        person_clothes_right_sleeve_mask = person_clothes_right_sleeve_mask.cuda()
        person_clothes_mask_concate = person_clothes_mask_concate.cuda()
        person_clothes_left_sleeve = person_clothes * person_clothes_left_sleeve_mask
        person_clothes_torso = person_clothes * person_clothes_torso_mask
        person_clothes_right_sleeve = person_clothes * person_clothes_right_sleeve_mask

        preserve_mask = data['preserve_mask'].cuda()
        preserve_mask2 = data['preserve_mask2'].cuda()
        preserve_mask3 = data['preserve_mask3'].cuda()

        # 模型的前向傳播
        densepose = F.interpolate(densepose, size=pose.shape[2:], mode='bilinear', align_corners=False)
        preserve_mask3 = F.interpolate(preserve_mask3, size=pose.shape[2:], mode='bilinear', align_corners=False)
        
        # densepose = F.interpolate(densepose, size=(256, 192), mode='bilinear', align_corners=False)
        # pose = F.interpolate(pose, size=(256, 192), mode='bilinear', align_corners=False)
        # preserve_mask3 = F.interpolate(preserve_mask3, size=(256, 192), mode='bilinear', align_corners=False)

        concat = torch.cat([densepose, pose, preserve_mask3], 1)        # 將DensePose、姿勢、遮罩拼接作為輸入

        # clothes = F.interpolate(clothes, size=(512, 384), mode='bilinear', align_corners=False)
        # pre_clothes_edge = F.interpolate(pre_clothes_edge, size=(512, 384), mode='bilinear', align_corners=False)
        # cloth_parse_for_d = F.interpolate(cloth_parse_for_d, size=(512, 384), mode='bilinear', align_corners=False)
        # clothes_left = F.interpolate(clothes_left, size=(512, 384), mode='bilinear', align_corners=False)
        # clothes_torso = F.interpolate(clothes_torso, size=(512, 384), mode='bilinear', align_corners=False)
        # clothes_right = F.interpolate(clothes_right, size=(512, 384), mode='bilinear', align_corners=False)
        # left_cloth_sleeve_mask = F.interpolate(left_cloth_sleeve_mask, size=(512, 384), mode='bilinear', align_corners=False)
        # cloth_torso_mask = F.interpolate(cloth_torso_mask, size=(512, 384), mode='bilinear', align_corners=False)
        # right_cloth_sleeve_mask = F.interpolate(right_cloth_sleeve_mask, size=(512, 384), mode='bilinear', align_corners=False)

        '''
        [batch size, channels, height, width]
        concat:  torch.Size([16, 51, 512, 384])
        clothes:  torch.Size([16, 3, 256, 192])
        pre_clothes_edge:  torch.Size([16, 1, 256, 192])
        cloth_parse_for_d:  torch.Size([16, 1, 256, 192])
        clothes_left:  torch.Size([16, 3, 256, 192])
        clothes_torso:  torch.Size([16, 3, 256, 192])
        clothes_right:  torch.Size([16, 3, 256, 192])
        left_cloth_sleeve_mask:  torch.Size([16, 1, 256, 192])
        cloth_torso_mask:  torch.Size([16, 1, 256, 192])
        right_cloth_sleeve_mask:  torch.Size([16, 1, 256, 192])
        preserve_mask3:  torch.Size([16, 1, 512, 384])
        image_concat:  torch.Size([48, 3, 256, 192])
        image_edge_concat:  torch.Size([48, 1, 256, 192])
        '''
        # print("concat: ", concat.shape)
        # print("clothes: ", clothes.shape)
        # print("pre_clothes_edge: ", pre_clothes_edge.shape)
        # print("cloth_parse_for_d: ", cloth_parse_for_d.shape)
        # print("clothes_left: ", clothes_left.shape)
        # print("clothes_torso: ", clothes_torso.shape)
        # print("clothes_right: ", clothes_right.shape)
        # print("left_cloth_sleeve_mask: ", left_cloth_sleeve_mask.shape)
        # print("cloth_torso_mask: ", cloth_torso_mask.shape)
        # print("right_cloth_sleeve_mask: ", right_cloth_sleeve_mask.shape)
        # print("preserve_mask3: ", preserve_mask3.shape)
        # print("test1")
        flow_out = model(concat, clothes, pre_clothes_edge, cloth_parse_for_d, \
                        clothes_left, clothes_torso, clothes_right, \
                        left_cloth_sleeve_mask, cloth_torso_mask, right_cloth_sleeve_mask, \
                        preserve_mask3)

        '''
        last_flow: 最終的變形場
        last_flow_all: 所有的變形場
        delta_list: 各層的細節變形
        seg_list: 生成的分割結果
        '''
        last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
            x_full_all, x_edge_full_all, attention_all, seg_list = flow_out

        # print("test2")
        # 判別器訓練
        set_requires_grad(discriminator, True)      # 開啟判別器參數更新 
        optimizer_D.zero_grad()
        pred_seg_D = seg_list[-1]                   # 使用模型最後一層的分割輸出
        D_concat = torch.cat([concat, cloth_parse_for_d.cuda()],1)
        D_in_fake = torch.cat([D_concat, pred_seg_D.detach()], 1)       # 判別器的假樣本（生成的分割結果）
        D_in_real = torch.cat([D_concat, seg_label_onehot_tensor], 1)   # 判別器的真樣本（標註分割結果）
        # print("test3")
        # 訓練判別器，使其區分真實和生成的分割結果
        loss_gan_D = (criterionLSGANloss(discriminator(
            D_in_fake), False) + criterionLSGANloss(discriminator(D_in_real), True)) * 0.5 * 0.1
        loss_gan_D.backward()
        optimizer_D.step()
        set_requires_grad(discriminator, False)     # 關閉判別器參數更新 
        # print("test4")
        D_in_fake_G = torch.cat([D_concat, pred_seg_D], 1)
        loss_gan_G = criterionLSGANloss(
            discriminator(D_in_fake_G), True)* 0.1

        bz = pose.size(0)

        epsilon = 0.001
        loss_smooth = 0
        # 使用 TV（Total Variation）損失，確保變形場 delta_list 平滑，避免劇烈變化
        loss_smooth = sum([AFWM.TVLoss(x*part_mask) for x in delta_list])

        # 計算其他損失
        loss_all = 0
        loss_l1_total = 0
        loss_vgg_total = 0
        loss_edge_total = 0
        loss_second_smooth_total = 0

        loss_full_l1_total = 0
        loss_full_vgg_total = 0
        loss_full_edge_total = 0

        loss_attention_total = 0
        loss_seg_ce_total = 0

        softmax = torch.nn.Softmax(dim=1)
        class_weight  = torch.FloatTensor([1,40,5,40,30,30,40]).cuda()
        criterionCE = nn.CrossEntropyLoss(weight=class_weight)
        # print("test5")
        # 進行 5 個不同層級的計算，每層的解析度逐步降低
        for num in range(5):
            # print("for ", num)
            # mode = 'nearest' 鄰插值，mode = 'bilinear' 雙線性插值
            cur_seg_label_tensor = F.interpolate(seg_label_tensor, scale_factor=0.5**(4-num), mode='nearest').cuda()

            pred_seg = seg_list[num]
            loss_seg_ce = criterionCE(pred_seg, cur_seg_label_tensor.long()[:,0,...])   # cross entropy
            # !!!
            pred_attention = attention_all[num]
            pred_mask_concate = torch.cat([pred_attention[:,0:1,...],pred_attention[:,1:2,...],pred_attention[:,2:3,...]],0)
            cur_person_clothes_mask_gt = F.interpolate(person_clothes_mask_concate, scale_factor=0.5**(4-num), mode='bilinear')
            loss_attention = criterionL1(pred_mask_concate,cur_person_clothes_mask_gt)  # Attention Loss

            # 局部衣物損失 (Local Clothes Loss)，局部影像、局部遮罩 !!!
            cur_person_clothes_left_sleeve = F.interpolate(person_clothes_left_sleeve, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_left_sleeve_mask = F.interpolate(person_clothes_left_sleeve_mask, scale_factor=0.5**(4-num), mode='bilinear')
            
            cur_person_clothes_torso = F.interpolate(person_clothes_torso, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_torso_mask = F.interpolate(person_clothes_torso_mask, scale_factor=0.5**(4-num), mode='bilinear')
            
            cur_person_clothes_right_sleeve = F.interpolate(person_clothes_right_sleeve, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_right_sleeve_mask = F.interpolate(person_clothes_right_sleeve_mask, scale_factor=0.5**(4-num), mode='bilinear')

            # 將局部衣物影像與對應遮罩按垂直方向拼接成整體特徵 !!!
            cur_person_clothes = torch.cat([cur_person_clothes_left_sleeve, cur_person_clothes_torso, cur_person_clothes_right_sleeve],0)
            cur_person_clothes_edge = torch.cat([cur_person_clothes_left_sleeve_mask, cur_person_clothes_torso_mask, cur_person_clothes_right_sleeve_mask],0)

            pred_clothes = x_all[num]    # !!
            pred_edge = x_edge_all[num]  # !!

            # 兩種不同遮罩進行合併，最終生成用於裁切的遮罩 !!!
            cur_preserve_mask = F.interpolate(preserve_mask, scale_factor=0.5**(4-num), mode='bilinear')
            cur_preserve_mask2 = F.interpolate(preserve_mask2, scale_factor=0.5**(4-num), mode='bilinear')
            cur_preserve_mask_concate = torch.cat([cur_preserve_mask,cur_preserve_mask2,cur_preserve_mask],0)

            # 創建全0遮罩 !!
            zero_mask = torch.zeros_like(cur_preserve_mask)
            # 衣物遮罩的合併 !!
            cur_person_clothes_mask_concate = torch.cat([cur_person_clothes_torso_mask,cur_person_clothes_left_sleeve_mask+cur_person_clothes_right_sleeve_mask,cur_person_clothes_torso_mask],0)
            # 遮罩合併與閾值化!!!
            cur_preserve_mask_concate += cur_person_clothes_mask_concate
            cur_preserve_mask_concate = (cur_preserve_mask_concate>0).float()
            
            # print("原pred_clothes: ", pred_clothes.shape)

            # 若當前訓練的 epoch 超過設定的 mask_epoch，則對預測結果進行裁切 !!!
            if epoch > opt.mask_epoch:
                pred_clothes = pred_clothes * (1-cur_preserve_mask_concate)
                pred_edge = pred_edge * (1-cur_preserve_mask_concate)
            
            '''
            pred_clothes.shape: [12, 3, height, width]
            cur_person_clothes.shape:   [12, 3, height, width]
            part_mask.shape:    [12, 1, 1, 1]
            '''
            # print("pred_clothes: ", pred_clothes.shape)
            # print("cur_person_clothes: ", cur_person_clothes.shape)

            # 預測結果與真實影像的損失 !!!
            loss_l1 = criterionL1(pred_clothes*part_mask, cur_person_clothes*part_mask)
            loss_vgg = criterionVGG(pred_clothes*part_mask, cur_person_clothes*part_mask)   #([12, 3, height, width], [12, 3, height, width])
            loss_edge = criterionL1(pred_edge*part_mask, cur_person_clothes_edge*part_mask)

            # 將真實衣物和邊緣圖像縮放至對應層級的版本!!!
            cur_person_clothes_full = F.interpolate(person_clothes, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_edge_full = F.interpolate(person_clothes_edge, scale_factor=0.5**(4-num), mode='bilinear')

            # 從預測結果中取得的衣物和邊緣預測圖像 !!!
            pred_clothes_full = x_full_all[num]
            pred_edge_full = x_edge_full_all[num]

            # 若超過閾值，將預測圖像與遮罩相乘，避免對某些區域進行更新 !!!
            if epoch > opt.mask_epoch:
                pred_clothes_full = pred_clothes_full * (1-cur_preserve_mask2)
                pred_edge_full = pred_edge_full * (1-cur_preserve_mask2)

            # 計算L1損失
            loss_full_l1 = criterionL1(pred_clothes_full, cur_person_clothes_full)
            loss_full_edge = criterionL1(pred_edge_full, cur_person_clothes_edge_full)

            '''流場損失(x 和 y 方向)'''
            b, c, h, w = delta_x_all[num].shape

            loss_flow_x = (delta_x_all[num].pow(2) + epsilon*epsilon).pow(0.45)
            loss_flow_x = loss_flow_x * part_mask
            loss_flow_x = torch.sum(loss_flow_x[0:int(b/3)]) / (int(b/3)*c*h*w) + \
                        40 * torch.sum(loss_flow_x[int(b/3):int(b/3)*2]) / (int(b/3)*c*h*w) + \
                        torch.sum(loss_flow_x[int(b/3)*2:]) / (int(b/3)*c*h*w)
            loss_flow_x = loss_flow_x / 3

            loss_flow_y = (delta_y_all[num].pow(2) + epsilon*epsilon).pow(0.45)
            loss_flow_y = loss_flow_y * part_mask
            loss_flow_y = torch.sum(loss_flow_y[0:int(b/3)]) / (int(b/3)*c*h*w) + \
                        40 * torch.sum(loss_flow_y[int(b/3):int(b/3)*2]) / (int(b/3)*c*h*w) + \
                        torch.sum(loss_flow_y[int(b/3)*2:]) / (int(b/3)*c*h*w)
            loss_flow_y = loss_flow_y / 3
            loss_second_smooth = loss_flow_x + loss_flow_y

            # 總損失
            loss_all = loss_all + (num+1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + \
                (num+1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth + \
                (num+1) * loss_full_l1 + \
                (num+1) * 2 * loss_full_edge + \
                (num+1) * loss_attention * 0.5 + \
                (num+1) * loss_seg_ce * 0.5 #-------

            # 每項損失根據num層級累計到各項損失
            loss_l1_total += loss_l1 * (num + 1)
            loss_vgg_total += loss_vgg * (num + 1) * 0.2
            loss_edge_total += loss_edge * (num + 1) * 2
            loss_second_smooth_total += loss_second_smooth * (num + 1) * 6

            loss_full_l1_total += (num+1) * loss_full_l1
            loss_full_edge_total += (num+1) * 2 * loss_full_edge
            loss_attention_total += (num+1) * loss_attention * 0.5
            loss_seg_ce_total += loss_seg_ce * (num+1) * 0.5 #-------------

        loss_all = 0.1 * loss_smooth + loss_all + loss_gan_G #gan_G -------------------
        # print("test6")
        # with mlflow.start_run():
        #     if step % opt.write_loss_frep == 0:
        #         if opt.local_rank == 0:
        #             mlflow.log_metric('loss_all', loss_all, step)
        #             mlflow.log_metric('loss_l1', loss_l1_total, step)
        #             mlflow.log_metric('loss_vgg', loss_vgg_total, step)
        #             mlflow.log_metric('loss_edge', loss_edge_total, step)
        #             mlflow.log_metric('loss_second_smooth', loss_second_smooth_total, step)
        #             mlflow.log_metric('loss_smooth', loss_smooth * opt.first_order_smooth_weight, step)
        #             mlflow.log_metric('loss_full_l1', loss_full_l1_total, step)
        #             mlflow.log_metric('loss_full_edge', loss_full_edge_total, step)
        #             mlflow.log_metric('loss_attention', loss_attention_total, step)
        #             mlflow.log_metric('loss_seg_ce', loss_seg_ce_total, step)
        # if step % opt.write_loss_frep == 0:
        #     if opt.local_rank == 0:
        #         writer.add_scalar('loss_all', loss_all, step)
        #         writer.add_scalar('loss_l1', loss_l1_total, step)
        #         writer.add_scalar('loss_vgg', loss_vgg_total, step)
        #         writer.add_scalar('loss_edge', loss_edge_total, step)
        #         writer.add_scalar('loss_second_smooth', loss_second_smooth_total, step)
        #         writer.add_scalar('loss_smooth', loss_smooth * opt.first_order_smooth_weight, step)
        #         writer.add_scalar('loss_full_l1', loss_full_l1_total, step)
        #         writer.add_scalar('loss_full_edge', loss_full_edge_total, step)
        #         writer.add_scalar('loss_attention', loss_attention_total, step)
        #         writer.add_scalar('loss_seg_ce', loss_seg_ce_total, step)

        # print("test7")
        # 梯度更新
        optimizer_warp.zero_grad()
        loss_all.backward()     # 調整參數
        optimizer_warp.step()
        # print("test8")
        # 顯示結果和錯誤
        bz = real_image.size(0)
        warped_cloth = x_all[4]
        left_warped_cloth = warped_cloth[0:bz]
        torso_warped_cloth = warped_cloth[bz:2*bz]
        right_warped_cloth = warped_cloth[2*bz:]
        # print("test9")
        # 將各部件合併
        warped_cloth = left_warped_cloth + torso_warped_cloth + right_warped_cloth
        # 變形衣物的完整圖像
        warped_cloth_full = x_full_all[-1]
        # print("test10")
        if result_out:
            if opt.local_rank == 0:
                # print("in if")
                # 不同的圖像或張量，代表不同的處理過程中的數據
                a = real_image.float().cuda()
                b = person_clothes.cuda()
                c = clothes.cuda()
                d = torch.cat([densepose_fore.cuda(),densepose_fore.cuda(),densepose_fore.cuda()],1)
                cm = cloth_parse_vis.cuda()
                e = warped_cloth

                bz = pose.size(0)
                # 對最終分割圖像進行softmax 計算並取最大值，標註像素最可能的類別 -----------------
                # change SGM model
                seg_preds = torch.argmax(softmax(seg_list[-1]),dim=1)[:,None,...].float()
                # 生成相對應的遮罩
                left_mask = (seg_preds==1).float()
                torso_mask = (seg_preds==2).float()
                right_mask = (seg_preds==3).float()

                # 將不同部件融合
                warped_cloth_fusion = left_warped_cloth * left_mask + \
                                    torso_warped_cloth * torso_mask + \
                                    right_warped_cloth * right_mask
                warped_cloth_fusion = warped_cloth_fusion *  (1-preserve_mask)

                # 融合後的衣物
                eee = warped_cloth_fusion

                # 進一步對各區域進行加權
                fused_attention = attention_all[-1]
                left_atten = fused_attention[:,0:1,...]
                torso_atten = fused_attention[:,1:2,...]
                right_atten = fused_attention[:,2:3,...]

                # 將大於0的部分標註為1，並在通道維度上複製以生成多通道圖像
                vis_pose = (pose > 0).float()
                vis_pose = torch.sum(vis_pose.cuda(), dim=1).unsqueeze(1)
                g = torch.cat([vis_pose, vis_pose, vis_pose], 1)

                h = torch.cat([preserve_mask, preserve_mask, preserve_mask], 1)
                h2 = torch.cat([preserve_mask2, preserve_mask2, preserve_mask2], 1)
                #------------
                seg_gt_vis = torch.cat([seg_gt_tensor,seg_gt_tensor,seg_gt_tensor],1).cuda()
                seg_preds = torch.argmax(softmax(seg_list[-1]),dim=1)[:,None,...].float()
                seg_preds = seg_preds / 6 * 2 - 1

                seg_preds_vis = torch.cat([seg_preds,seg_preds,seg_preds],1)
                
                # 圖像合併
                combine = torch.cat([a[0], c[0], cm[0], g[0], d[0], h[0], right_warped_cloth[0], torso_warped_cloth[0], \
                                    left_warped_cloth[0], e[0], eee[0], b[0], seg_preds_vis[0], seg_gt_vis[0]], 2).squeeze()

                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1)/2
                writer.add_image('combine', (combine.data + 1) / 2.0, step)
                rgb = (cv_img*255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite('sample/train_warping/result3/' +str(epoch).zfill(3)+'_'+str(step)+'.jpg', bgr)
                result_out = False

                # cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1)/2
                # # writer.add_image('combine', (combine.data + 1) / 2.0, step)
                # rgb = (cv_img*255).astype(np.uint8)
                # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmpfile:
                #     # 使用 OpenCV 保存圖像
                #     cv2.imwrite('sample/'+opt.name+'/' + str(epoch).zfill(3)+'_'+str(step)+'.jpg', bgr)
                    
                #     # 在 MLflow 中記錄圖像作為 artifact
                #     mlflow.log_artifact(tmpfile.name, 'images')
                # mlflow.end_run()
        

        step += 1
        # 計算時間差
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        # 計算預計剩餘時間
        step_delta = (step_per_batch-step % step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
        eta = iter_delta_time*step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        # 獲取當前時間戳
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        # 訓練過程中的輸出信息
        if step % opt.print_freq == 0:
            if opt.local_rank == 0:
                print('{}:{}:[step-{}]--[loss-{:.6f}]--[learning rate-{}]--[ETA-{}]'.format(now, epoch_iter, step, loss_all, model.old_lr, eta))
        # print("test11")
        # 結束訓練
        if epoch_iter >= dataset_size:
            break
    
    # 計算並打印訓練時間
    iter_end_time = time.time()
    if opt.local_rank == 0:
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # 每隔一定步數保存模型
    if epoch % opt.save_epoch_freq == 0:
        if opt.local_rank == 0:
            # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            save_checkpoint(model, os.path.join(opt.checkpoints_dir, 'new_' + opt.name, 'new1_PBAFN_train_warping_epoch_%03d.pth' % (epoch+1)))
            save_checkpoint(discriminator, os.path.join(opt.checkpoints_dir, 'new_' + opt.name, 'new_1PBAFN_train_warping_D_epoch_%03d.pth' % (epoch+1)))

    # 學習率更新
    if epoch > opt.niter:
        model.update_learning_rate(optimizer_warp)
        discriminator.update_learning_rate(optimizer_D, opt)

