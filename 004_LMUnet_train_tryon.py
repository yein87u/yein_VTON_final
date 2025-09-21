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
from models.LightMUNet import LightMUNet
from models.networks import ResUnetGenerator, SpectralDiscriminator, load_checkpoint_parallel, VGGLoss, GANLoss, set_requires_grad, save_checkpoint
from monai.utils import UpsampleMode
from came_pytorch import CAME
# from lightning.fabric.strategies.fsdp import fsdp_overlap_step_with_backward ##
def CreateDataset(opt):
    dataset = aligned_dataset.AlignedDataset()
    dataset.initialize(opt, mode="train")
    
    return dataset

import torchvision

# 清理未使用的 CUDA 記憶體
torch.cuda.empty_cache()

# 檢查 PyTorch 是否安裝成功
print("PyTorch version:", torch.__version__)

# 檢查 CUDA 是否可用
print("CUDA Available:", torch.cuda.is_available())

# 確認 CUDA 版本
print("CUDA Version:", torch.version.cuda)
print(torchvision.__version__)

opt = TrainOptions().parse()
opt.warproot = '/mnt/c/Users/User/Desktop/yein_VTON/sample/test_warping/result4/train'
opt.segroot = '/mnt/c/Users/User/Desktop/yein_VTON/sample/test_warping/seg4/train'

run_path = 'runs/train_tryon/LMUnet/'+opt.name         # 'runs/flow'
sample_path = 'sample/train_tryon/LMUnet/'+opt.name    # 'sample/flow'
os.makedirs(run_path, exist_ok=True)
os.makedirs(sample_path, exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

# 設定GPU編號
torch.cuda.set_device(opt.local_rank)
# 用GPU跑
device = torch.device(f'cuda:{opt.local_rank}')

train_data = CreateDataset(opt)     # 11632 不是 11647是因為batchsize分割時data沒辦法被整除
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataset_size = len(train_loader)    # 727

gen_model = LightMUNet(
    spatial_dims=2,          # 2D U-Net
    init_filters=8,         # 與 ngf=64 相同
    in_channels=36,          # 輸入通道數
    out_channels=4,          # 輸出通道數
    blocks_down= (1, 1, 1, 1),
    blocks_up= (1, 1, 1),
)
gen_model.train()
gen_model.cuda()
gen_model.to(device)

# opt.PBAFN_gen_LMUnet_checkpoint = '/mnt/c/Users/User/Desktop/yein_VTON/checkpoints/flow/PBAFN_tryon_gen_LMUnet_epoch_072.pth'
# opt.pretrain_checkpoint_D = '/mnt/c/Users/User/Desktop/yein_VTON/checkpoints/flow/PBAFN_tryon_D_LMUnet_epoch_072.pth' 

if opt.PBAFN_gen_LMUnet_checkpoint is not None:
    load_checkpoint_parallel(gen_model, opt.PBAFN_gen_LMUnet_checkpoint)

# if opt.isTrain and len(opt.gpu_ids):
#     model_gen = torch.nn.parallel.DistributedDataParallel(gen_model, device_ids=[opt.local_rank])

# params_gen = [p for p in model_gen.parameters()]
optimizer_gen = torch.optim.Adam(gen_model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizer_gen = CAME(gen_model.parameters(), 
#                      lr = 1e-4,
#                      weight_decay=1e-4,
#                      betas=(opt.beta1, 0.999, 0.9999),
#                      eps=(1e-8, 1e-16))

discriminator = SpectralDiscriminator(opt, input_nc=39, ndf=64, n_layers=3,
                                      norm_layer=functools.partial(nn.BatchNorm2d, 
                                      affine=True, track_running_stats=True), use_sigmoid=False)
discriminator.train()
discriminator.cuda()

# 讓模型在訓練或推理時使用之前訓練過的參數
if opt.pretrain_checkpoint_D is not None:
    load_checkpoint_parallel(discriminator, opt.pretrain_checkpoint_D)

# 只有一張GPU只需要將模型移至GPU上進行訓練就好了
discriminator = discriminator.to(device)

# 定義並初始化優化器，用來訓練鑑別器，lambda 條件用來過濾那些標記為需要梯度計算的參數，parameters 只會返回可訓練參數
params_D = list(filter(lambda p: p.requires_grad, discriminator.parameters()))

# 初始化 Adam 優化器
optimizer_D = torch.optim.Adam(params_D, lr=opt.lr_D, betas=(opt.beta1, 0.999))
# optimizer_D = CAME(params_D, 
#                     lr = 2e-4,
#                     weight_decay=1e-4,
#                     betas=(opt.beta1, 0.999, 0.9999),
#                     eps=(1e-8, 1e-16))

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()
criterionLSGANloss = GANLoss().cuda()

if opt.local_rank == 0:
    writer = SummaryWriter(run_path)
    print('#training images = %d' % dataset_size)

start_epoch, epoch_iter = 1, 0
total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size

# optimizers = [optimizer_gen([p], ...) for p in gen_model.parameters()]

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    torch.cuda.empty_cache()
    result_out = True
    epoch_start_time = time.time()  # 紀錄每次epoch開始時間
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(tqdm(train_loader, desc="Training Tryon", total=len(train_loader), ncols=100)):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1

        person_clothes_edge = data['person_clothes_mask'].to(device)
        real_image = data['image'].to(device)
        preserve_mask = data['preserve_mask3'].to(device)
        preserve_region = real_image * preserve_mask
        warped_cloth = data['warped_cloth'].to(device)
        warped_prod_edge = data['warped_edge'].to(device)
        arms_color = data['arms_color'].to(device)
        arms_neck_label= data['arms_neck_lable'].to(device)
        pose = data['pose'].to(device)
        background_color = data['background_color'].to(device)
        cloth_id = data['cloth_id']
        person_id = data['person_id']

        # merge = warped_cloth + preserve_region
        # preserve_region_img = (merge.permute(1, 2, 0).detach().cpu().numpy()+1)/2
        # preserve_region_img_rgb = (preserve_region_img*255).astype(np.uint8)
        # preserve_region_img_bgr = cv2.cvtColor(preserve_region_img_rgb, cv2.COLOR_RGB2BGR)
        # os.makedirs('/mnt/c/Users/User/Desktop/yein_VTON/sample/checkimage/', exist_ok=True)
        # cv2.imwrite('/mnt/c/Users/User/Desktop/yein_VTON/sample/checkimage/'+ person_id[0] +'.jpg', preserve_region_img_bgr)

        gen_inputs = torch.cat([preserve_region, warped_cloth, warped_prod_edge, arms_neck_label, arms_color, pose], 1)
        # gen_inputs = torch.cat([merge, warped_prod_edge, arms_neck_label, arms_color], 1)

        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite1 = m_composite * warped_prod_edge
        m_composite =  person_clothes_edge.cuda()*m_composite1
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        # p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        # p_tryon = (warped_cloth * warped_prod_edge) + (p_rendered * (1-warped_prod_edge))

        set_requires_grad(discriminator, True)
        optimizer_D.zero_grad()
        pred_seg_D = p_rendered
        D_in_fake = torch.cat([gen_inputs, pred_seg_D.detach()], 1)
        D_in_real = torch.cat([gen_inputs, real_image], 1)
        loss_gan_D = (criterionLSGANloss(discriminator(
            D_in_fake), False) + criterionLSGANloss(discriminator(D_in_real), True)) * 0.5
        loss_gan_D.backward()
        optimizer_D.step()
        set_requires_grad(discriminator, False)

        D_in_fake_G = torch.cat([gen_inputs, pred_seg_D], 1)
        loss_gan_G = criterionLSGANloss(discriminator(D_in_fake_G), True)* 0.5

        loss_mask_l1 = torch.mean(torch.abs(1 - m_composite)) * 5
        loss_l1 = criterionL1(p_tryon, real_image.cuda())
        loss_vgg = criterionVGG(p_tryon,real_image.cuda())
        bg_loss_l1 = criterionL1(p_rendered, real_image.cuda())
        bg_loss_vgg = criterionVGG(p_rendered, real_image.cuda())
        gen_loss = (loss_l1 + loss_vgg + bg_loss_l1 + bg_loss_vgg + loss_mask_l1)

        if step % opt.write_loss_frep == 0:
            if opt.local_rank == 0:
                writer.add_scalar('gen_loss', gen_loss, step)
                writer.add_scalar('gen_mask_l1_loss', loss_mask_l1 * 1.0, step)
                writer.add_scalar('gen_l1_loss', loss_l1 * 5, step)
                writer.add_scalar('gen_vgg_loss', loss_vgg, step)
                writer.add_scalar('gen_bg_l1_loss', bg_loss_l1 * 5, step)
                writer.add_scalar('gen_bg_vgg_loss', bg_loss_vgg, step)
                writer.add_scalar('gen_GAN_G_loss', loss_gan_G, step)
                writer.add_scalar('gen_GAN_D_loss', loss_gan_D, step)

        loss_all =  gen_loss + loss_gan_G

        optimizer_gen.zero_grad()
        loss_all.backward()
        optimizer_gen.step()


        ############## Display results and errors ##########
        if result_out:
            a = real_image.float().cuda()
            e = warped_cloth
            f = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
            ff = arms_color
            g = preserve_region.cuda()
            # g = merge
            vis_pose = (pose > 0).float()
            vis_pose = torch.sum(vis_pose.cuda(), dim=1).unsqueeze(1)
            vis_pose_mask = (vis_pose > 0).to(
                vis_pose.device).to(vis_pose.dtype)
            h = torch.cat([vis_pose, vis_pose, vis_pose], 1)
            i = p_rendered
            j = torch.cat([m_composite1, m_composite1, m_composite1], 1)
            k = p_tryon

            l = torch.cat([arms_neck_label,arms_neck_label,arms_neck_label],1)

            combine = torch.cat([a[0], h[0], g[0], f[0], l[0], ff[0], e[0], j[0], i[0], k[0]], 2).squeeze()
            # combine = torch.cat([a[0], g[0], f[0], l[0], ff[0], e[0], j[0], i[0], k[0]], 2).squeeze()
            cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1)/2
            writer.add_image('combine', (combine.data + 1) / 2.0, step)
            rgb = (cv_img*255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            os.makedirs('sample/train_tryon/LMUnet/result8/', exist_ok=True)
            cv2.imwrite('sample/train_tryon/LMUnet/result8/'+str(epoch) + '_'+str(step)+'.jpg', bgr)
            result_out = False

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch-step%step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
        eta = iter_delta_time*step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')

        if step % opt.print_freq == 0:
            if opt.local_rank == 0:
                print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(
                    now, epoch_iter, step, loss_all, eta))

        if epoch_iter >= dataset_size:
            break

    iter_end_time = time.time()
    if opt.local_rank == 0:
      print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
      if opt.local_rank == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        save_checkpoint(gen_model, os.path.join(opt.checkpoints_dir, "new_"+ opt.name, 'PBAFN_tryon_gen_LMUnet8_epoch_%03d.pth' % (epoch)))
        save_checkpoint(discriminator, os.path.join(opt.checkpoints_dir, "new_" + opt.name, 'PBAFN_tryon_D_LMUnet8_epoch_%03d.pth' % (epoch)))
    if epoch > opt.niter:
        discriminator.update_learning_rate(optimizer_D)
        gen_model.update_learning_rate(optimizer_gen)

