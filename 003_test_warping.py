import cv2
# print(cv2.__version__)
from options.train_options import TrainOptions
from models.networks import load_checkpoint_parallel
from models.AFWM import AFWM_Vitonhd_lrarms
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

opt = TrainOptions().parse()
os.makedirs('sample/test_warping/'+opt.name, exist_ok=True)

def CreateDataset(opt):
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    dataset.initialize(opt, mode='train')

    return dataset

torch.cuda.set_device(opt.local_rank)

device = torch.device(f'cuda:{opt.local_rank}')



# opt.PBAFN_warp_checkpoint = 'C:\\Users\\User\\Downloads\\GP-VTON_Chackpoint\\gp-vton_gen_vitonhd_wskin_wgan_lrarms_1029\\PBAFN_warp_epoch_071.pth'
# opt.pretrain_checkpoint_D = 'C:\\Users\\User\\Downloads\\GP-VTON_Chackpoint\\gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027\\PBAFN_D_epoch_071.pth'

opt.PBAFN_warp_checkpoint = '/mnt/c/Users/User/Desktop/yein_VTON/checkpoints/new_flow/PBAFN_train_warping_epoch_025.pth'

test_data = CreateDataset(opt)
test_loader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, num_workers=0, pin_memory=True)

warp_model = AFWM_Vitonhd_lrarms(opt, 51)

warp_model.eval()
warp_model.cuda()
load_checkpoint_parallel(warp_model, opt.PBAFN_warp_checkpoint)

softmax = torch.nn.Softmax(dim=1)

model = warp_model.to(device)

for ii, data in enumerate(tqdm.tqdm(test_loader)):
    with torch.no_grad():   # 執行推論時暫時停用梯度計算
        pre_clothes_edge = data['edge']
        clothes = data['cloth']
        clothes = clothes * pre_clothes_edge
        pose = data['pose']

        size = data['cloth'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.zeros(oneHot_size1, dtype=torch.float32, device='cuda')
        data['densepose'] = (data['densepose'] / 229.0 * 24.0).round().long()
        densepose = densepose.scatter_(1,data['densepose'].data.long().cuda(),1.0)
        densepose = densepose * 2.0 - 1.0
        densepose_fore = data['densepose']/24.0

        left_cloth_sleeve_mask = data['flat_clothes_left_mask']
        cloth_torso_mask = data['flat_clothes_middle_mask']
        right_cloth_sleeve_mask = data['flat_clothes_right_mask']

        clothes_left = clothes * left_cloth_sleeve_mask
        clothes_torso = clothes * cloth_torso_mask
        clothes_right = clothes * right_cloth_sleeve_mask

        cloth_parse_for_d = data['flat_clothes_label'].cuda()
        pose = pose.cuda()
        clothes = clothes.cuda()
        clothes_left = clothes_left.cuda()
        clothes_torso = clothes_torso.cuda()
        clothes_right = clothes_right.cuda()
        pre_clothes_edge = pre_clothes_edge.cuda()
        left_cloth_sleeve_mask = left_cloth_sleeve_mask.cuda()
        cloth_torso_mask = cloth_torso_mask.cuda()
        right_cloth_sleeve_mask = right_cloth_sleeve_mask.cuda()
        preserve_mask3 = data['preserve_mask3'].cuda()


        if opt.resolution == 512:
            concat = torch.cat([densepose, pose, preserve_mask3], 1)
            
            # print('concat: ', concat.shape)
            # print('clothes: ', clothes.shape)
            # print('pre_clothes_edge: ', pre_clothes_edge.shape)
            # print('cloth_parse_for_d: ', cloth_parse_for_d.shape)
            # print('clothes_left: ', clothes_left.shape)
            # print('clothes_torso: ', clothes_torso.shape)
            # print('clothes_right: ', clothes_right.shape)
            # print('left_cloth_sleeve_mask: ', left_cloth_sleeve_mask.shape)
            # print('cloth_torso_mask: ', cloth_torso_mask.shape)
            # print('right_cloth_sleeve_mask: ', right_cloth_sleeve_mask.shape)
            # print('preserve_mask3: ', preserve_mask3.shape)
            
            # print('concat: ', concat.unique())
            # print('clothes: ', clothes.unique())
            # print('pre_clothes_edge: ', pre_clothes_edge.unique())
            # print('cloth_parse_for_d: ', cloth_parse_for_d.unique())
            # print('clothes_left: ', clothes_left.unique())
            # print('clothes_torso: ', clothes_torso.unique())
            # print('clothes_right: ', clothes_right.unique())
            # print('left_cloth_sleeve_mask: ', left_cloth_sleeve_mask.unique())
            # print('cloth_torso_mask: ', cloth_torso_mask.unique())
            # print('right_cloth_sleeve_mask: ', right_cloth_sleeve_mask.unique())
            # print('preserve_mask3: ', preserve_mask3.unique())

            

            # pre_clothes_edge = (pre_clothes_edge*2)-1
            # cloth_parse_for_d = (cloth_parse_for_d*2)-1
            # left_cloth_sleeve_mask = (left_cloth_sleeve_mask*2)-1
            # cloth_torso_mask = (cloth_torso_mask*2)-1
            # right_cloth_sleeve_mask = (right_cloth_sleeve_mask*2)-1
            # preserve_mask3 = (preserve_mask3*2)-1

            
            flow_out = model(concat, clothes, pre_clothes_edge, cloth_parse_for_d, \
                                clothes_left, clothes_torso, clothes_right, \
                                left_cloth_sleeve_mask, cloth_torso_mask, right_cloth_sleeve_mask, \
                                preserve_mask3)

            last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
                x_full_all, x_edge_full_all, attention_all, seg_list = flow_out

        bz = pose.size(0)

        left_last_flow = last_flow[0:bz]
        torso_last_flow = last_flow[bz:2*bz]
        right_last_flow = last_flow[2*bz:]

        left_warped_full_cloth = F.grid_sample(clothes_left.cuda(), left_last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='zeros')
        torso_warped_full_cloth = F.grid_sample(clothes_torso.cuda(), torso_last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='zeros')
        right_warped_full_cloth = F.grid_sample(clothes_right.cuda(), right_last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='zeros')

        left_warped_cloth_edge = F.grid_sample(left_cloth_sleeve_mask.cuda(), left_last_flow.permute(0, 2, 3, 1),mode='nearest', padding_mode='zeros')
        torso_warped_cloth_edge = F.grid_sample(cloth_torso_mask.cuda(), torso_last_flow.permute(0, 2, 3, 1),mode='nearest', padding_mode='zeros')
        right_warped_cloth_edge = F.grid_sample(right_cloth_sleeve_mask.cuda(), right_last_flow.permute(0, 2, 3, 1),mode='nearest', padding_mode='zeros')

        for bb in range(bz):
            seg_preds = torch.argmax(softmax(seg_list[-1]),dim=1)[:,None,...].float()
            # print("seg_preds.unique()", seg_preds.unique())
            # for kkk in range(len(seg_preds.unique())):
            #     img = (seg_preds[bb]==kkk).float()
            #     img = (img.cpu().numpy() * 255).astype(np.uint8)
            #     img = img.squeeze(0)
                
            #     # img = cv2.cvtColor(img, cv2.COLOR_GRAY)
            #     os.makedirs('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\seg_preds_mask', exist_ok = True)
            #     cv2.imwrite(f'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\seg_preds_mask\\{kkk}.png', img)


            # c_type = data['c_type'][bb]
            # change SGM
            left_mask = (seg_preds[bb]==1).float()
            torso_mask = (seg_preds[bb]==2).float()
            right_mask = (seg_preds[bb]==3).float()

            left_arm_mask = (seg_preds[bb]==4).float()
            right_arm_mask = (seg_preds[bb]==5).float()
            neck_mask = (seg_preds[bb]==6).float()

            warped_cloth_fusion = left_warped_full_cloth[bb] * left_mask + \
                                torso_warped_full_cloth[bb] * torso_mask + \
                                right_warped_full_cloth[bb] * right_mask
            
            warped_edge_fusion = left_warped_cloth_edge[bb] * left_mask * 1 + \
                                    torso_warped_cloth_edge[bb] * torso_mask * 2 + \
                                    right_warped_cloth_edge[bb] * right_mask * 3

            warped_cloth_fusion = warped_cloth_fusion * (1-preserve_mask3[bb])
            warped_edge_fusion = warped_edge_fusion * (1-preserve_mask3[bb])
            
            warped_edge_fusion = warped_edge_fusion + \
                                    left_arm_mask * 4 + \
                                    right_arm_mask * 5 + \
                                    neck_mask * 6
            # left_right_sleeve = left_warped_full_cloth[bb] + right_warped_full_cloth[bb]
            # warped_cloth_fusion = torch.where(torso_mask > 0, torso_warped_full_cloth[bb], left_right_sleeve)
            # left_right_mask = (left_warped_cloth_edge[bb] * 1) + (right_warped_cloth_edge[bb] * 3)
            # warped_edge_fusion = torch.where(torso_mask > 0, torso_warped_cloth_edge[bb]*2, left_right_mask)
            # left_right_neck = left_arm_mask * 4 + right_arm_mask * 5 + neck_mask * 6
            # warped_edge_fusion = torch.where(warped_edge_fusion > 0, warped_edge_fusion, left_right_neck)

            eee = warped_cloth_fusion
            eee_edge = torch.cat([warped_edge_fusion,warped_edge_fusion,warped_edge_fusion],0)
            eee_edge = eee_edge.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

            cv_img = (eee.permute(1, 2, 0).detach().cpu().numpy()+1)/2
            rgb = (cv_img*255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # bgr = np.concatenate([bgr,eee_edge],1)------------------------------------

            # seg_preds = torch.argmax(softmax(seg_list[-1]),dim=1)[:,None,...].float()
            # seg_preds = seg_preds / 6 * 2 - 1
            seg_preds = warped_edge_fusion / 6 * 2 - 1
            seg_preds_vis = torch.cat([seg_preds,seg_preds,seg_preds],0)
            # print(seg_preds_vis.shape)
            
            # seg = (seg_preds_vis[bb].permute(1, 2, 0).detach().cpu().numpy()+1)/2
            seg = (seg_preds_vis.permute(1, 2, 0).detach().cpu().numpy()+1)/2
            seg = (seg*255).astype(np.uint8)
            # print(np.unique(seg))

            seg[seg == 42] = 21  # 左袖
            seg[seg == 85] = 5  # 軀幹
            seg[seg == 127] = 22  # 右袖
            seg[seg == 170] = 15  # 左臂
            seg[seg == 212] = 16  # 右臂
            seg[seg == 255] = 11  # 脖子

            # 建立 3 通道的 RGB 圖像
            # seg = seg[..., 0]
            # seg_colored = np.zeros((*seg.shape, 3), dtype=np.uint8)

            # # 定義顏色映射 (B, G, R 格式)
            # color_map = {
            #     21: (255, 0, 0),    # 左袖 - 紅色
            #     5: (0, 255, 0),     # 軀幹 - 綠色
            #     22: (0, 0, 255),    # 右袖 - 藍色
            #     15: (255, 255, 0),  # 左臂 - 青色
            #     16: (255, 0, 255),  # 右臂 - 洋紅色
            #     11: (0, 255, 255)   # 脖子 - 黃色
            # }

            # # 遍歷標籤並填充顏色
            # for label, color in color_map.items():
            #     seg_colored[seg == label] = color  # 直接賦值 BGR 顏色
            
            seg_save = cv2.cvtColor(seg, cv2.COLOR_RGB2BGR)

            # # 保留區域
            # parse_ttp_pred = data['parse_ttp_pred']
            # print(parse_ttp_pred[bb].shape)
            # parse_ttp_pred_1 = (parse_ttp_pred[bb].detach().cpu().numpy()+1)/2
            # parse_ttp_pred_1 = (parse_ttp_pred_1).astype(np.uint8)
            # print(np.unique(parse_ttp_pred_1))
            
            '''
            116 right sleeve
            119 right arm
            245 left sleeve
            248 left arm
            250 neck
            253 torso
            '''
            # parsing_np_1[parsing_np_1 == 118] = 18  # right leg?
            # parsing_np_1[parsing_np_1 == 120] = 14  # face
            # parsing_np_1[parsing_np_1 == 126] = 2  # hair
            # parsing_np_1[parsing_np_1 == 127] = 0  # background
            # parsing_np_1[parsing_np_1 == 247] = 17  # left leg?
            # parsing_np_1[parsing_np_1 == 249] = 13  # skirts
            # parsing_np_1[parsing_np_1 == 122] = 10  # right pants
            # parsing_np_1[parsing_np_1 == 251] = 9  # left pants
            
            # parsing_np_1[parsing_np_1 == 116] = 0  # right leg?
            # parsing_np_1[parsing_np_1 == 119] = 0  # face
            # parsing_np_1[parsing_np_1 == 245] = 0  # hair
            # parsing_np_1[parsing_np_1 == 248] = 0  # background
            # parsing_np_1[parsing_np_1 == 250] = 0  # left leg?
            # parsing_np_1[parsing_np_1 == 253] = 0  # skirts
            
            # ptexttp_save = cv2.cvtColor(parse_ttp_pred_1, cv2.COLOR_RGB2BGR)

            cloth_id = data['cloth_path'][bb].split('/')[-1]
            person_id = data['img_path'][bb].split('/')[-1]
            # cloth_id = data['cloth_path'][bb].split('\\')[-1]
            # person_id = data['img_path'][bb].split('\\')[-1]
            # 儲存結果
            os.makedirs('sample/test_warping/result4/train/', exist_ok = True)
            save_path = 'sample/test_warping/result4/train/'+person_id+'___'+cloth_id[:-4]+'.png'
            cv2.imwrite(save_path, bgr)

            os.makedirs('sample/test_warping/seg4/train/', exist_ok = True)
            save_path = 'sample/test_warping/seg4/train/'+person_id+'___'+cloth_id[:-4]+'.png'
            cv2.imwrite(save_path, seg_save)
            # /mnt/Users/User/Desktop/yein_VTON/sample/test_warping/seg2/test/
            # os.makedirs('sample/test_warping/ptexttp/test/', exist_ok = True)
            # save_path = 'sample/test_warping/ptexttp/test/'+person_id+'___'+cloth_id[:-4]+'.png'
            # cv2.imwrite(save_path, ptexttp_save)

            # print('cloth_parse_for_d: ', pre_clothes_edge[bb].shape)
            # pm = pre_clothes_edge[bb].permute(1, 2, 0)
            # # print('cl: ', cl.shape)

            # img_pre_clothes_edge = (pm.cpu().numpy()*255).astype(np.uint8)
            # # img_pre_clothes_edge = cv2.cvtColor(img_pre_clothes_edge, cv2.COLOR_RGB2BGR)
            # img_pre_clothes_edge = cv2.applyColorMap(img_pre_clothes_edge, cv2.COLORMAP_JET)
            # os.makedirs('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\pre_clothes_edge', exist_ok = True)
            # cv2.imwrite(f'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\pre_clothes_edge\\{bb}.png', img_pre_clothes_edge)
            

