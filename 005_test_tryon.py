from options.train_options import TrainOptions
from models.networks import ResUnetGenerator, load_checkpoint_parallel
import torch.nn as nn
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
from tqdm import tqdm


# from utils.fid_scores import fid_pytorch
from lpips import LPIPS
from datetime import datetime
import torch_fidelity
from torch.autograd import Variable
from datetime import datetime
from torch.nn import functional as F
from math import exp
from models.LightMUNet import LightMUNet
from models.CMUnet import VSSM

opt = TrainOptions().parse()
os.makedirs('sample/test_tryon/'+opt.name, exist_ok=True)

def CreateDataset(opt):
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    dataset.initialize(opt, mode='test')

    return dataset

torch.cuda.set_device(opt.local_rank)

device = torch.device(f'cuda:{opt.local_rank}')

'''
GP_ResUNet
LMUnet
CMUnet
'''
model_choose = 'LMUnet'


if(model_choose == 'GP_ResUNet'):
    opt.PBAFN_gen_checkpoint = '/mnt/c/Users/User/Desktop/yein_VTON/checkpoints/flow/PBAFN_tryon_gen_epoch_039.pth'
    gen_model = ResUnetGenerator(36, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d, opt=opt)
elif(model_choose == 'LMUnet'):
    opt.PBAFN_gen_checkpoint = '/mnt/c/Users/User/Desktop/yein_VTON/checkpoints/new_flow/PBAFN_tryon_gen_LMUnet8_epoch_050.pth'
    gen_model = LightMUNet(
        spatial_dims=2,          # 2D U-Net
        init_filters=8,         # 與 ngf=64 相同
        in_channels=36,          # 輸入通道數
        out_channels=4,          # 輸出通道數
        blocks_down=[1, 1, 1, 1],
        blocks_up=[1, 1, 1],
    )
elif(model_choose == 'CMUnet'):
    opt.PBAFN_gen_checkpoint = '/mnt/c/Users/User/Desktop/yein_VTON/checkpoints/flow/PBAFN_tryon_gen_CMUnet_new_epoch_003.pth'
    gen_model = VSSM().to('cuda')
else:
    print("please choose model")

opt.warproot = '/mnt/c/Users/User/Desktop/yein_VTON/sample/test_warping/result4/test'
opt.segroot = '/mnt/c/Users/User/Desktop/yein_VTON/sample/test_warping/seg4/test'

test_data = CreateDataset(opt)
test_loader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, num_workers=0, pin_memory=True)

# gen_model = ResUnetGenerator(36, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d, opt=opt)

gen_model.cuda()
load_checkpoint_parallel(gen_model, opt.PBAFN_gen_checkpoint)

gen_model = gen_model.to(device)
gen_model.eval()

with torch.no_grad():
    for ii, data in enumerate(tqdm(test_loader)):
        real_image = data['image'].cuda()
        clothes = data['cloth'].cuda()
        preserve_mask = data['preserve_mask3'].cuda()
        preserve_region = real_image * preserve_mask
        warped_cloth = data['warped_cloth'].cuda()
        warped_prod_edge = data['warped_edge'].cuda()
        arms_color = data['arms_color'].cuda()
        arms_neck_label= data['arms_neck_lable'].cuda()
        pose = data['pose'].cuda()
        real_image = data['image'].cuda()
        background_color = data['background_color'].cuda()

        # merge = warped_cloth + preserve_region
        gen_inputs = torch.cat([preserve_region, warped_cloth, warped_prod_edge, arms_neck_label, arms_color, pose], 1)
        # gen_inputs = torch.cat([merge, warped_prod_edge, arms_neck_label, arms_color], 1)

        gen_outputs = gen_model(gen_inputs)

        # p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)

        # p_rendered = torch.tanh(p_rendered)
        # m_composite = torch.sigmoid(m_composite)
        # warped_cloth_mask = (warped_cloth > 0.2).float()
        # m_composite = m_composite * warped_cloth_mask
        # preserve_rendered = p_rendered * (1 - m_composite)
        # # 獲取背景顏色並填補空缺
        # filled_background = background_color * m_composite
        # preserve_rendered += filled_background 
        # p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        # k = p_tryon
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_prod_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        k = p_tryon


        
        # seg_path = 'C:/Users/User/Desktop/yein_VTON/sample/checkdata/seg/'+person_id[0]+'___'+cloth_id[0][:-4]+'.png'
        # os.makedirs('C:/Users/User/Desktop/yein_VTON/sample/checkdata/result/', exist_ok = True)

        bz = pose.size(0)
        for bb in range(bz):
            cloth_id = data['cloth_id'][bb]
            person_id = data['person_id'][bb]
            combine = k[bb].squeeze()
        
            cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1)/2
            rgb = (cv_img*255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            save_path = f'/mnt/c/Users/User/Desktop/yein_VTON/sample/test_tryon/resultv8/{model_choose}/'+person_id+'___'+cloth_id[:-4]+'.png'
            path_fid = f'/mnt/c/Users/User/Desktop/yein_VTON/sample/test_tryon/fake_images8/{model_choose}/'+person_id
            os.makedirs(f'/mnt/c/Users/User/Desktop/yein_VTON/sample/test_tryon/resultv8/{model_choose}/', exist_ok = True)
            os.makedirs(f'/mnt/c/Users/User/Desktop/yein_VTON/sample/test_tryon/fake_images8/{model_choose}/', exist_ok = True)

            cv2.imwrite(save_path, bgr)
            cv2.imwrite(path_fid, bgr)


