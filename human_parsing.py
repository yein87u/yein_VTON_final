

# import os
# import torch
# import functools
# import torch.nn as nn
# import time
# import torch.nn.functional as F
# import cv2
# import datetime
# import numpy as np
# from tqdm import tqdm
# from options.train_options import TrainOptions
# from data import aligned_dataset
# from torch.utils.data import DataLoader
# from torch.utils.checkpoint import checkpoint
# from torch.utils.tensorboard import SummaryWriter
# from models import AFWM
# from models.networks import SpectralDiscriminator, load_checkpoint_parallel, VGGLoss, GANLoss, set_requires_grad, save_checkpoint
# from models.LightMUNet import LightMUNet

# def CreateDataset(opt):
#     dataset = aligned_dataset.AlignedDataset()
#     dataset.initialize(opt)
    
#     return dataset

# opt = TrainOptions().parse()

# torch.cuda.empty_cache()

# # 設定GPU編號
# torch.cuda.set_device(opt.local_rank)

# # 用GPU跑
# device = torch.device(f'cuda:{opt.local_rank}')

# train_data = CreateDataset(opt)     # 11632 不是 11647是因為batchsize分割時data沒辦法被整除
# train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, num_workers=0, pin_memory=True)
# dataset_size = len(train_loader)    # 727
# # print("train_data len:", len(train_data))

# parsing_model = LightMUNet(
#     spatial_dims=2,          # 2D U-Net
#     init_filters=8,         # 與 ngf=64 相同
#     in_channels=3,          # 輸入通道數
#     out_channels=1,          # 輸出通道數
#     blocks_down=[1, 1, 1, 1],
#     blocks_up=[1, 1, 1],
# )
# parsing_model.train()

# if opt.PBAFN_human_parsing_checkpoint is not None:
#     load_checkpoint_parallel(parsing_model, opt.PBAFN_human_parsing_checkpoint)
# # warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

# # 將模型移動到GPU上
# model = parsing_model.to(device)

# '''與主模型有關'''
# # 提取可訓練參數
# params_parsing = [p for p in model.parameters()]   # 遍歷所有參數
# optimizer_parsing = torch.optim.Adam(params_parsing, lr=opt.lr, betas=(opt.beta1, 0.999)) # 設置Adam優化器，設置betas控制一階、二階動量估計平滑程度

# # 判別器模型，做譜歸一化，控制訓練時的梯度，讓模型更穩定
# discriminator = SpectralDiscriminator(opt, input_nc=1, ndf=64, n_layers=3, # 輸入影像層數、卷積層的初始通道數、鑑別器的卷積層數
#                                     # 正規化層使用 InstanceNorm2d，affine 允許使用學習的縮放和平移參數，track_running_stats 允許追蹤均值和變異數
#                                     norm_layer=functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True), 
#                                     use_sigmoid=False)  # 不允許使用Sigmoid激活函數，在對抗網路中不需要將輸出限制在[0, 1]

# discriminator.train()
# discriminator.cuda()

# # 讓模型在訓練或推理時使用之前訓練過的參數
# if opt.pretrain_checkpoint_D is not None:
#     load_checkpoint_parallel(discriminator, opt.pretrain_checkpoint_D)

# # 只有一張GPU只需要將模型移至GPU上進行訓練就好了
# discriminator = discriminator.to(device)

# '''與判別器有關'''
# # 定義並初始化優化器，用來訓練鑑別器，lambda 條件用來過濾那些標記為需要梯度計算的參數，parameters 只會返回可訓練參數
# params_D = list(filter(lambda p: p.requires_grad, discriminator.parameters()))

# # 初始化 Adam 優化器
# optimizer_D = torch.optim.Adam(params_D, lr=opt.lr_D, betas=(opt.beta1, 0.999))

# criterionL1 = nn.L1Loss()               # 平均誤差損失(MAE)
# criterionVGG = VGGLoss()
# criterionLSGANloss = GANLoss().cuda()
# softmax = torch.nn.Softmax(dim=1)       # dim表示對每一行計算softmax，使所有元素都在[0, 1]之間

# start_epoch = 1 # 起始epoch
# epoch_iter = 0  # 訓練了幾次epoch
# total_steps = (start_epoch-1) * dataset_size + epoch_iter   # 總步數
# step = 0    # 初始步數
# step_per_batch = dataset_size   # epoch 中的步數（batch size）

# for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
#     result_out = True
#     epoch_start_time = time.time()  # 紀錄每次epoch開始時間
#     if epoch != start_epoch:
#         epoch_iter = epoch_iter % dataset_size
    
#     for i, data in enumerate(tqdm(train_loader, desc="Training_parsing", total=len(train_loader))):
#         iter_start_time = time.time()

#         total_steps += 1
#         epoch_iter += 1
#         person_id = data['person_id']
#         people_image = data['image'].to(device)                              # 人物的原始影像
#         parsing_np = data['parsing_np']
#         if parsing_np.dtype == torch.uint8:
#             parsing_np = parsing_np.permute(0, 3, 1, 2).float() / 255.0
#         parsing_np = parsing_np.to(device)

#         result_parsing = model(people_image)

#         set_requires_grad(discriminator, True)      # 開啟判別器參數更新 
#         optimizer_D.zero_grad()
#         D_in_fake = result_parsing.detach()       # 判別器的假樣本（生成的分割結果）
#         D_in_real = parsing_np   # 判別器的真樣本（標註分割結果）
#         # print("test3")
#         # 訓練判別器，使其區分真實和生成的分割結果
#         loss_gan_D = (criterionLSGANloss(discriminator(D_in_fake), False) + 
#                       criterionLSGANloss(discriminator(D_in_real), True)) * 0.5
#         loss_gan_D.backward()
#         optimizer_D.step()
#         set_requires_grad(discriminator, False)     # 關閉判別器參數更新 
#         loss_gan_G = criterionLSGANloss(discriminator(result_parsing), True)

#         loss_l1 = criterionL1(result_parsing, parsing_np.cuda())    # fake, real
#         # loss_vgg = criterionVGG(result_parsing, parsing_np.cuda())  # fake, real

#         loss_all = loss_l1 + loss_gan_G
#         optimizer_parsing.zero_grad()
#         loss_all.backward()
#         optimizer_parsing.step()

#         if result_out:
#             bz = 1
#             save_path = '/mnt/c/Users/User/Desktop/yein_VTON/sample/human_parsing/train/'
#             os.makedirs(save_path, exist_ok = True)
#             result_parsing_img = result_parsing[bz][0].detach().cpu().numpy()
#             result_parsing_img = (result_parsing_img * 255).clip(0, 255).astype(np.uint8)

#             cv2.imwrite(os.path.join(save_path, f'{epoch}_' + str(person_id[bz])), result_parsing_img)
#             result_out = False
#     step += 1
#     # 計算時間差
#     iter_end_time = time.time()
#     iter_delta_time = iter_end_time - iter_start_time
#     # 計算預計剩餘時間
#     step_delta = (step_per_batch-step % step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
#     eta = iter_delta_time*step_delta
#     eta = str(datetime.timedelta(seconds=int(eta)))
#     # 獲取當前時間戳
#     time_stamp = datetime.datetime.now()
#     now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
#     # 訓練過程中的輸出信息
#     if step % opt.print_freq == 0:
#         if opt.local_rank == 0:
#             print('{}:{}:[step-{}]--[loss-{:.6f}]--[learning rate-{}]--[ETA-{}]'.format(now, epoch_iter, step, loss_all, model.old_lr, eta))
#     # print("test11")

#     # 計算並打印訓練時間
#     iter_end_time = time.time()
#     if opt.local_rank == 0:
#         print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

#     # 每隔一定步數保存模型
#     if epoch % opt.save_epoch_freq == 0:
#         if opt.local_rank == 0:
#             # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
#             save_checkpoint(model, os.path.join(opt.checkpoints_dir, 'human_parsing', 'train_human_parsing_epoch_%03d.pth' % (epoch+1)))
#             save_checkpoint(discriminator, os.path.join(opt.checkpoints_dir, 'human_parsing', 'train_human_parsing_D_epoch_%03d.pth' % (epoch+1)))

#     # 學習率更新
#     if epoch > opt.niter:
#         model.update_learning_rate(optimizer_parsing)
#         discriminator.update_learning_rate(optimizer_D, opt)

        







