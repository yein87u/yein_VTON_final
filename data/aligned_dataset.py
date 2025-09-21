import os
import pickle
import math
import numpy as np
import cv2
import torch
import random
import json
import torch.nn.functional as F
import pycocotools.mask as maskUtils
from PIL import Image, ImageDraw
from data.base_dataset import BaseDataset, get_params, get_transform
from random import random
import torchvision.transforms as transforms
from tqdm import tqdm

class AlignedDataset():
    def initialize(self, opt, mode = 'train'):
        self.opt = opt
        self.root = opt.dataroot
        self.warproot = opt.warproot
        self.segroot = opt.segroot
        self.predict = opt.predict
        self.image_size = opt.image_size    # 在訓練、推論期間的圖片大小(512)
        self.mode = mode
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))])
        self.countTest = 0

        # 處理影像尺寸
        if self.image_size == 512:  # this
            self.fine_height=512
            self.fine_width=384
            self.radius=2
        else:
            self.fine_height=1024
            self.fine_width=768
            self.radius=16

        with open(opt.image_all_data, 'rb') as f:
            self.lines = pickle.load(f, encoding='UTF-8')
        
        self.dataset_size = len(self.lines[mode])
        # print(self.lines['train'][0])

        self.P_paths = []
        self.C_paths = []
        self.C_Mask_paths = []
        self.C_Componet_Mask_paths = []
        self.Pose_paths = []
        self.Dense_paths = []
        self.Parse_paths = []

        for line in self.lines[mode]:
            self.P_paths.append(line['images_dir'])
            self.C_paths.append(line['cloth_dir'])
            self.C_Mask_paths.append(line['cloth_mask_dir'])
            self.C_Componet_Mask_paths.append(line['cloth_component_mask_dir'])
            self.Pose_paths.append(line['keypoints_dir'])
            self.Dense_paths.append(line['image_densepose_dir'])
            self.Parse_paths.append(line['parse_dir'])
              
        ratio_dict = None
        if self.mode == 'train':
            ratio_dict = {}
            person_clothes_ratio_txt = os.path.join(self.root, 'VITON-HD1024_Origin/ratio_upper_train.txt')
            with open(person_clothes_ratio_txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                c_name, ratio = line.strip().split()
                ratio = float(ratio)
                ratio_dict[c_name] = ratio
        self.ratio_dict = ratio_dict
        # print(ratio_dict)
            

        # print(self.P_paths[0])
        # print(self.C_paths[0])
        # print(self.C_Mask_paths[0])
        # print(self.C_Componet_Mask_paths[0])
        # print(self.Pose_paths[0])
        # print(self.Dense_paths[0])
        # print(self.Parse_paths[0])

        # print("P_paths", len(self.P_paths))
        # print("C_paths", len(self.C_paths))
        # print("C_Mask_paths", len(self.C_Mask_paths))
        # print("C_Componet_Mask_paths", len(self.C_Componet_Mask_paths))
        # print("Pose_paths", len(self.Pose_paths))
        # print("Dense_paths", len(self.Dense_paths))
        # print("Parse_paths", len(self.Parse_paths))

        # pass the ratio of person and clothes save

    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[..., np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps = [x1, y1, x2, y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / \
            (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / \
            (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1 < cos2:
            kps.extend([x3, y3, x4, y4])
        else:
            kps.extend([x4, y4, x3, y3])

        kps = np.array(kps).reshape(1, -1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask

    def get_hand_mask(self, hand_keypoints, h, w):
        # 獲取關鍵點(x, y, 信心值)
        s_x, s_y, s_c = hand_keypoints[0]   #肩膀
        e_x, e_y, e_c = hand_keypoints[1]   #肘部
        w_x, w_y, w_c = hand_keypoints[2]   #手腕
        s_x, s_y = s_x/2, s_y/2   #肩膀
        e_x, e_y = e_x/2, e_y/2   #肘部
        w_x, w_y = w_x/2, w_y/2   #手腕

        # 初始化上半與下半部遮罩
        up_mask = np.ones((h, w, 1), dtype=np.float32)
        bottom_mask = np.ones((h, w, 1), dtype=np.float32)

        # 上半部遮罩
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            if self.image_size == 512:
                kernel = np.ones((50, 50), np.uint8)
            else:
                kernel = np.ones((100, 100), np.uint8)
            up_mask = cv2.dilate(up_mask, kernel, iterations=1) # 使用膨脹操作擴展上半部遮罩，使遮罩邊界便平滑
            up_mask = (up_mask > 0).astype(np.float32)[..., np.newaxis]
        
        #下半部遮罩
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            if self.image_size == 512:
                kernel = np.ones((30, 30), np.uint8)
            else:
                kernel = np.ones((60, 60), np.uint8)
            bottom_mask = cv2.dilate(bottom_mask, kernel, iterations=1) # 使用膨脹操作擴展下半部遮罩，使遮罩邊界便平滑
            bottom_mask = (bottom_mask > 0).astype(np.float32)[..., np.newaxis]

        return up_mask, bottom_mask #返回遮罩

    #獲取手掌遮罩, (基本遮罩, 上半部手部遮罩, 下半部手部遮罩)
    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)    # 取得基本遮罩與上半部的交集
        hand_mask = hand_mask - inter_up_mask   # 去掉手臂上半部
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)== 2).astype(np.float32)   #取得基本遮罩與下半部的交集
        palm_mask = hand_mask - inter_bottom_mask   #去掉手臂下半部
        return palm_mask    #得到手掌遮罩

    def get_palm(self, parsing, keypoints):
        h, w = parsing.shape[0:2]
        # 取得手部關鍵點
        left_hand_keypoints = keypoints[[5, 6, 7], :].copy()
        right_hand_keypoints = keypoints[[2, 3, 4], :].copy()

        # 取得手部遮罩(包含手指頭)
        left_hand_up_mask, left_hand_bottom_mask = self.get_hand_mask(left_hand_keypoints, h, w)
        right_hand_up_mask, right_hand_bottom_mask = self.get_hand_mask(right_hand_keypoints, h, w)

        # 創建左右手基本遮罩，並且對應區域設為1，反之設為0
        left_hand_mask = (parsing == 15).astype(np.float32)
        right_hand_mask = (parsing == 16).astype(np.float32)

        # 取得手掌遮罩，(基本遮罩, 上半部手部遮罩, 下半部手部遮罩)
        left_palm_mask = self.get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_bottom_mask)
        right_palm_mask = self.get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_bottom_mask)
        
        # 合併左右手遮罩
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask    # 回傳二維np陣列，內容是標記手掌的位置

    def __getitem__(self, index):
        ### person image
        # print(f"index type: {type(index)}")
        '''CheckData'''
        cloth_id = self.C_paths[index].split('/')[-1]
        person_id = self.P_paths[index].split('/')[-1]
        # cloth_id = self.C_paths[index].split('\\')[-1]
        # person_id = self.P_paths[index].split('\\')[-1]
        # print("person_id: ", person_id, "cloth_id", cloth_id)
        array_path = [self.P_paths[index], self.C_paths[index], self.C_Mask_paths[index],
                      self.C_Componet_Mask_paths[index], self.Dense_paths[index], self.Parse_paths[index]]
        # for iii, path in enumerate(array_path):
        #     img = Image.open(path).convert('RGB')
        #     img = np.array(img)
        #     os.makedirs('sample/test_warping/CheckData/', exist_ok = True)
        #     save_path = 'sample/test_warping/CheckData/'+person_id+'___'+cloth_id[:-4] +'___'+ str(iii) +'.png'
        #     cv2.imwrite(save_path, img)


        P_path = self.P_paths[index]

        P = Image.open(P_path).convert('RGB')
        P_np = np.array(P)
        params = get_params(self.opt, P.size)   # 取得、切割圖片
        transform_for_rgb = get_transform(self.opt, params) #圖片轉張量
        P_tensor = transform_for_rgb(P) # 將轉換過後的張量存起來
        '''[3, 256, 192] => [3, 512, 384]'''

        ### person 2d pose
        # 一維的關鍵點數據重塑為 N 行 3 列的格式，其中每行代表一個關鍵點的 (x座標, y座標, 信心值)
        with open(self.Pose_paths[index], 'r') as file:
            pd = json.load(file)
        pose_data = np.array(pd['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        # pose_data = np.array(self.Pose_paths[index]['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        point_num = pose_data.shape[0]  #返回pose_data的行數
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)    #定義姿勢圖的大小(特徵點數量, 高, 寬)
        r = self.radius # 每個關鍵點的半徑，表示每個關鍵點的影響範圍
        im_pose = Image.new('L', (self.fine_width, self.fine_height))   #創建新的灰階圖像, 'L'代表灰階
        pose_draw = ImageDraw.Draw(im_pose) #創建繪製實例，使pose_draw能在im_pose進行繪製
        for i in range(point_num):
            # 再宣告一個畫布，使每畫一個關鍵點都是獨立的，不會被其他關鍵點干擾
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)

            # 獲取第i個關鍵點的x, y
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            pointx /= 2
            pointy /= 2

            # 確保關鍵點的坐標有效
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')  #在 one_map中x, y座標畫長方形，半徑為r
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white') # 在im_pose中x, y座標畫長方形，半徑為r
            
            one_map = transform_for_rgb(one_map.convert('RGB')) # 將one_map轉換成RGB格式後再用transform_for_rgb轉換成張量
            one_map_resized = F.interpolate(one_map[0].unsqueeze(0).unsqueeze(0), size=(512, 384), mode='bilinear', align_corners=False)
            # 刪除多餘的維度，得到 [512, 384] 的二維張量
            pose_map[i] = one_map_resized.squeeze(0).squeeze(0)
            # pose_map[i] = one_map[0]    # 轉換完的張量為(1, fine_height, fine_width)，因此拿取[0]，就會是二維的灰階資料
        Pose_tensor = pose_map  # 存入
        
        '''[25, 512, 384]'''

        # point_line = Image.new('L', (self.fine_width, self.fine_height))
        # draw = ImageDraw.Draw(point_line)
        # RShoulder_x, RShoulder_y = pose_data[2,0]/2, pose_data[2,1]/2
        # LShoulder_x, LShoulder_y = pose_data[5,0]/2, pose_data[5,1]/2
        # RElbow_x, RElbow_y = pose_data[3,0]/2, pose_data[3,1]/2
        # LElbow_x, LElbow_y = pose_data[6,0]/2, pose_data[6,1]/2
        # RWrist_x, RWrist_y = pose_data[4,0]/2, pose_data[4,1]/2
        # LWrist_x, LWrist_y = pose_data[7,0]/2, pose_data[7,1]/2

        # if RElbow_x>1 and RElbow_y>1 and RWrist_x>1 and RWrist_y>1:
        #     draw.line((RElbow_x, RElbow_y, RWrist_x, RWrist_y), 'white')
        # if LElbow_x>1 and LElbow_y>1 and LWrist_x>1 and LWrist_y>1:
        #     draw.line((LElbow_x, LElbow_y, LWrist_x, LWrist_y), 'white')

        # if RElbow_x>1 and RElbow_y>1 and RShoulder_x>1 and RShoulder_y>1:
        #     draw.line((RElbow_x, RElbow_y, RShoulder_x, RShoulder_y), 'white')
        # if LElbow_x>1 and LElbow_y>1 and LShoulder_x>1 and LShoulder_y>1:
        #     draw.line((LElbow_x, LElbow_y, LShoulder_x, LShoulder_y), 'white')


        # point_line = self.transform(point_line) 

        # print(point_line.unique())
        # print('point_line.unique(): ', point_line.unique())
        # img_point_line = (point_line.permute(1, 2, 0) + 1) / 2
        # img_point_line = ((img_point_line * 255).cpu().numpy()).astype(np.uint8)
        # img_point_line = cv2.cvtColor(img_point_line, cv2.COLOR_RGB2BGR)
        # os.makedirs('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\train_SGM\\point_line', exist_ok = True)
        # cv2.imwrite(f'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\train_SGM\\point_line\\{person_id}___{cloth_id}.png', img_point_line)

        ### person 3d pose, 用於處理姿勢估計和圖像分割
        dense_mask = Image.open(self.Dense_paths[index]).convert('L')   #用灰階圖片打開圖片
        transform_for_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) #獲取轉換的動作
        dense_mask_tensor = transform_for_mask(dense_mask) * 255.0  #依照取得的動作轉換成張量
        dense_mask_tensor = dense_mask_tensor[0:1, ...] # 對張量進行裁剪，使其形狀變為(1, height, width)
        '''[1, 256, 192] => [1, 512, 384]'''

        ### person parsing
        parsing = Image.open(self.Parse_paths[index]).convert('L')  #讀取圖片
        parsing_tensor = transform_for_mask(parsing) * 255.0    # 將圖片轉為張量
        parsing_np = (parsing_tensor.numpy().transpose(1, 2, 0)[..., 0:1]).astype(np.uint8) #將張量轉為np陣列，並調整維度順序(height, width, channel)
        palm_mask_np = self.get_palm(parsing_np, pose_data)  #去掉手臂上下部分，獲取手掌遮罩
        '''(256, 192, 1) => (512, 384, 1)'''

        # 生成各部位遮罩
        person_clothes_left_sleeve_mask_np = (parsing_np==21).astype(int) + (parsing_np==24).astype(int) + (parsing_np==26).astype(int)    # 左袖
        person_clothes_torso_mask_np = (parsing_np==5).astype(int) + (parsing_np==6).astype(int) + (parsing_np==7).astype(int)   #上裝與洋裝
        person_clothes_right_sleeve_mask_np = (parsing_np==22).astype(int) + (parsing_np==25).astype(int) + (parsing_np==27).astype(int)   #右袖
        person_clothes_mask_np = person_clothes_left_sleeve_mask_np + person_clothes_torso_mask_np + person_clothes_right_sleeve_mask_np    #依照人體取得的衣服遮罩
        left_arm_mask_np = (parsing_np==15).astype(int) # 左臂
        right_arm_mask_np = (parsing_np==16).astype(int)    # 右臂
        hand_mask_np = (parsing_np==15).astype(int) + (parsing_np==16).astype(int)  #手臂遮罩
        neck_mask_np = (parsing_np==11).astype(int) # 脖子

        # 將各個部位從np陣列轉為torch張量，以便模型使用
        person_clothes_left_sleeve_mask_tensor = torch.tensor(person_clothes_left_sleeve_mask_np.transpose(2, 0, 1)).float()
        person_clothes_torso_mask_tensor = torch.tensor(person_clothes_torso_mask_np.transpose(2, 0, 1)).float()
        person_clothes_right_sleeve_mask_tensor = torch.tensor(person_clothes_right_sleeve_mask_np.transpose(2, 0, 1)).float()
        person_clothes_mask_tensor = torch.tensor(person_clothes_mask_np.transpose(2, 0, 1)).float()
        left_arm_mask_tensor = torch.tensor(left_arm_mask_np.transpose(2, 0, 1)).float()
        right_arm_mask_tensor = torch.tensor(right_arm_mask_np.transpose(2, 0, 1)).float()
        neck_mask_tensor = torch.tensor(neck_mask_np.transpose(2, 0, 1)).float()
        '''all [1, 256, 192]'''

        #生成語意分割張量，每個部位對應不一樣的數字
        seg_gt_tensor = person_clothes_left_sleeve_mask_tensor * 1 + person_clothes_torso_mask_tensor * 2 + \
                        person_clothes_right_sleeve_mask_tensor * 3 + left_arm_mask_tensor * 4 + \
                        right_arm_mask_tensor * 5 + neck_mask_tensor * 6
        '''[1, 256, 192] => [1, 512, 384]'''

        #生成背景遮罩張量
        background_mask_tensor = 1 - (person_clothes_left_sleeve_mask_tensor + person_clothes_torso_mask_tensor + \
                                    person_clothes_right_sleeve_mask_tensor + left_arm_mask_tensor + right_arm_mask_tensor + \
                                    neck_mask_tensor)
        '''[1, 256, 192]'''
        
        #在樣本數量上進行拼接，將多個樣本合併成一個大的批量，因此第二個參數設為0
        seg_gt_onehot_tensor = torch.cat([background_mask_tensor, person_clothes_left_sleeve_mask_tensor, \
                                        person_clothes_torso_mask_tensor, person_clothes_right_sleeve_mask_tensor, \
                                        left_arm_mask_tensor, right_arm_mask_tensor, neck_mask_tensor],0)
        '''[7, 256, 192] => [7, 512, 384]'''

        im_parse = torch.from_numpy(parsing_np)
        
        # if self.segroot or self.predict:
            # 保留區域 0, 1, 2, 3, 4, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 28
            # parse_ttp = (parsing_np == 0).astype(np.float32) + \
            #         (parsing_np == 1).astype(np.float32) + \
            #         (parsing_np == 2).astype(np.float32) + \
            #         (parsing_np == 3).astype(np.float32) + \
            #         (parsing_np == 4).astype(np.float32) + \
            #         (parsing_np == 8).astype(np.float32) + \
            #         (parsing_np == 9).astype(np.float32) + \
            #         (parsing_np == 10).astype(np.float32) + \
            #         (parsing_np == 12).astype(np.float32) + \
            #         (parsing_np == 13).astype(np.float32) + \
            #         (parsing_np == 14).astype(np.float32) + \
            #         (parsing_np == 17).astype(np.float32) + \
            #         (parsing_np == 18).astype(np.float32) + \
            #         (parsing_np == 19).astype(np.float32) + \
            #         (parsing_np == 20).astype(np.float32) + \
            #         (parsing_np == 23).astype(np.float32) + \
            #         (parsing_np == 28).astype(np.float32)
            # ptexttp = torch.from_numpy(parse_ttp)

            # 生成語意區域 5, 6, 7, 11, 15, 16, 21, 22, 24, 25, 26, 27
            # parse_cloth_arm_neck = (parsing_np == 5).astype(np.float32) + \
            #         (parsing_np == 6).astype(np.float32) + \
            #         (parsing_np == 7).astype(np.float32) + \
            #         (parsing_np == 11).astype(np.float32) + \
            #         (parsing_np == 15).astype(np.float32) + \
            #         (parsing_np == 16).astype(np.float32) + \
            #         (parsing_np == 21).astype(np.float32) + \
            #         (parsing_np == 22).astype(np.float32) + \
            #         (parsing_np == 24).astype(np.float32) + \
            #         (parsing_np == 25).astype(np.float32) + \
            #         (parsing_np == 26).astype(np.float32) + \
            #         (parsing_np == 27).astype(np.float32)
            # can_bin = parse_cloth_arm_neck

            # parse_ttp_pred = parsing_np * (1 - parse_cloth_arm_neck)
            # ptexttp_dim_change = ptexttp.permute(2,0,1)

            # im_ttp = P_tensor * ptexttp_dim_change - (1- ptexttp_dim_change)

            # seg_shape = np.zeros((29, 512, 384), dtype=np.uint8) # 0~28, 共29

            # for hh in range(512):
            #     for ww in range(384):
            #         seg_shape[parsing_np[hh][ww][0]][hh][ww] = 1
            # seg_shape = torch.tensor(seg_shape, dtype=torch.float32)
            # seg_shape = seg_shape * 2 - 1   # 映射至[-1, 1]
    
            # for kkk in range(24):
            #     seg_shape[kkk] *= P_tensor
            # 測試seg_shape用
            # for kkk in range(24):
            #     img_kkk = (seg_shape[kkk].cpu().numpy() * 255).astype(np.uint8)
            #     # img_kkk = img_kkk.transpose(1, 2, 0)
            #     os.makedirs('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\seg\\', exist_ok = True)
            #     cv2.imwrite(f'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\seg\\{kkk}.png', img_kkk)

            # parse_arm = (parsing_np == 15).astype(np.float32) + \
            #         (parsing_np == 16).astype(np.float32)

            # pam = torch.from_numpy(parse_arm)
            # pam = pam.permute(2,0,1)
            # im_a = P_tensor * pam + (1 - pam) # [-1,1], fill 1 for other parts

        # hands_size = random.randint(10, 40)
        # hands_size = 30
            
        # arms_eliminate = Image.new('L', (self.fine_width, self.fine_height))
        # draw_arms_eliminate = ImageDraw.Draw(arms_eliminate)
        # pointrx = int(pose_data[4][0]) / 2
        # pointry = int(pose_data[4][1]) / 2
        # pointlx = int(pose_data[7][0]) / 2
        # pointly = int(pose_data[7][1]) / 2
        # # print(pose_data)
        # # print(f"pointlx: {pointlx}, pointly: {pointly}")
        # # print(f"pointrx: {pointrx}, pointry: {pointry}")    
        # if pointlx > 1 and pointly > 1:
        #     # print('point 1')
        #     draw_arms_eliminate.rectangle((pointlx - hands_size, pointly - hands_size, pointlx + hands_size, pointly + hands_size), fill = 255, outline = 255)
        # if pointrx > 1 and pointry > 1:
        #     # print('point 2')
        #     draw_arms_eliminate.rectangle((pointrx - hands_size, pointry - hands_size, pointrx + hands_size, pointry + hands_size), fill = 255, outline = 255)
        # arms_eliminate_np = np.array(arms_eliminate)
        # # print('np.unique(arms_eliminate_np): ', np.unique(arms_eliminate_np))
        # arms_eliminate = self.transform(arms_eliminate) # 變[-1, 1]
        # # print('arms_eliminate.shape: ', arms_eliminate.shape)
        # # print('arms_eliminate.unique: ', torch.unique(arms_eliminate))
        # # print('2arms_eliminate.unique(): ', arms_eliminate.unique())

        # part_arms = (arms_eliminate + 1) * 0.5  # 變[0, 1]
        # part_arms = part_arms*im_a + (1-part_arms)

        # arm_par = ((seg_shape[15,:,:] + 1) / 2) + ((seg_shape[16,:,:] + 1) / 2) # shape: [1, 512, 384]
        # # print('arm_par.unique: ', arm_par.unique())
        # part_arms_par = part_arms*arm_par

        
        # 印出測試part_arms_par
        # print('part_arms_par.unique(): ', part_arms_par.unique())
        # img_part_arms_par = (part_arms_par.permute(1, 2, 0) + 1) / 2
        # print('part_arms_par.unique(): ', part_arms_par.unique())
        # img_part_arms_par = ((img_part_arms_par * 255).cpu().numpy()).astype(np.uint8)
        # img_part_arms_par = cv2.cvtColor(img_part_arms_par, cv2.COLOR_RGB2BGR)
        # os.makedirs('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\train_tryon\\part_arms_par', exist_ok = True)
        # cv2.imwrite('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\train_tryon\\part_arms_par\\1.png', img_part_arms_par)

        # 印出測試part_arms
        # print('part_arms.unique(): ', part_arms.unique())
        # img_part_arms = (part_arms.permute(1, 2, 0) + 1) / 2
        # print('part_arms.unique(): ', part_arms.unique())
        # img_part_arms = ((img_part_arms * 255).cpu().numpy()).astype(np.uint8)
        # img_part_arms = cv2.cvtColor(img_part_arms, cv2.COLOR_RGB2BGR)
        # os.makedirs('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\train_tryon\\arms_eliminate', exist_ok = True)
        # cv2.imwrite('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\train_tryon\\arms_eliminate\\1.png', img_part_arms)

        # 印出測試im_a
        # print('im_a.unique(): ', im_a.unique())
        # img_im_a = (im_a.permute(1, 2, 0) + 1) / 2
        # print('im_a.unique(): ', im_a.unique())
        # img_im_a = ((img_im_a * 255).cpu().numpy()).astype(np.uint8)
        # img_im_a = cv2.cvtColor(img_im_a, cv2.COLOR_RGB2BGR)
        # os.makedirs('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\train_tryon\\im_a', exist_ok = True)
        # cv2.imwrite('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\train_tryon\\im_a\\1.png', img_im_a)


        ### preserve region mask        
        if self.opt.no_dynamic_mask or self.ratio_dict is None:
            preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
            preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
        else:
            s = self.C_paths[index].split('/')[-1][:-4]+'.png'
            pc_ratio = self.ratio_dict[s]
            if pc_ratio < 0.9:
                preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
            elif pc_ratio < 0.95:
                if random() < 0.5:
                    preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
                else:
                    preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,12,14,23,26,27]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
            else:
                if random() < 0.1:
                    preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
                else:
                    preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,12,14,23,26,27]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)

        ### preserve region mask, 保留區域的遮罩，沒有ratio資料，因此為靜態遮罩生成。
        # preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
        # preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)

        preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
        preserve_mask_np = np.sum(preserve_mask_np,axis=0)

        preserve_mask1_np = preserve_mask_for_loss_np + palm_mask_np
        preserve_mask2_np = preserve_mask_for_loss_np + hand_mask_np
        preserve_mask3_np = preserve_mask_np + palm_mask_np
        
        # print(np.unique(left_hand_up_mask))
        # seg_save = preserve_mask3_np*255
        # os.makedirs('sample/test_warping/seg4/preserve_mask3_np/', exist_ok = True)
        # save_path = f'sample/test_warping/seg4/preserve_mask3_np/{self.countTest}.png'
        # self.countTest+=1
        # cv2.imwrite(save_path, seg_save)
        # preserve_mask3_np = preserve_mask_for_loss_np + palm_mask_np

        preserve_mask1_tensor = torch.tensor(preserve_mask1_np.transpose(2,0,1)).float()
        '''[1, 256, 192] => [1, 512, 384]'''

        preserve_mask2_tensor = torch.tensor(preserve_mask2_np.transpose(2,0,1)).float()
        '''[1, 256, 192] => [1, 512, 384]'''
        preserve_mask3_tensor = torch.tensor(preserve_mask3_np.transpose(2,0,1)).float()
        '''
        img = Image.open(self.Parse_paths[index]).convert('L')  #讀取圖片
        mask_values = [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]   #欲保留區塊
        img_np = np.array(img)  # 將圖片轉換為 NumPy 陣列
        # 創建遮罩
        mask = np.isin(img_np, mask_values).astype(np.float32)
        preserve_mask_tensor = torch.tensor(mask).float()
        '''

        # 衣服影像
        C_path = self.C_paths[index]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_for_rgb(C) #圖片轉成張量
        '''[3, 256, 192] => [3, 512, 384]'''

        # 衣服遮罩
        CM = Image.open(self.C_Mask_paths[index]).convert('L')
        CM_tensor = transform_for_mask(CM)  # 依照取得的動作轉換成張量
        CM_tensor = torch.where(CM_tensor >= 0.5, torch.tensor(1.0), torch.tensor(0.0))

        '''[1, 256, 192] => [3, 512, 384]'''

        # 衣服部件標籤遮罩
        cloth_parsing = Image.open(self.C_Componet_Mask_paths[index]).convert('L')
        cloth_parsing_tensor = transform_for_mask(cloth_parsing) * 255.0    # 將圖片轉為張量
        cloth_parsing_tensor = cloth_parsing_tensor[0:1, ...]   # 對張量進行裁剪，使其形狀變為(1, height, width)

        cloth_parsing_np = (cloth_parsing_tensor.numpy().transpose(1,2,0)).astype(int)  #轉為np陣列並調整維度為(height, width, channel)
        flat_cloth_left_mask_np = (cloth_parsing_np==21).astype(int)    # 生成左側遮罩
        flat_cloth_middle_mask_np = (cloth_parsing_np==5).astype(int) + \
                                    (cloth_parsing_np==24).astype(int) + \
                                    (cloth_parsing_np==13).astype(int)  #生成中間遮罩
        flat_cloth_right_mask_np = (cloth_parsing_np==22).astype(int)   #生成右邊遮罩

        # 按左至右編號，並除3做標準化，使標籤數值在[0, 1]中
        flat_cloth_label_np = flat_cloth_left_mask_np * 1 + flat_cloth_middle_mask_np * 2 + flat_cloth_right_mask_np * 3
        flat_cloth_label_np = flat_cloth_label_np / 3
        
        #將pytorch張量轉為np陣列
        flat_cloth_left_mask_tensor = torch.tensor(flat_cloth_left_mask_np.transpose(2, 0, 1)).float()
        flat_cloth_middle_mask_tensor = torch.tensor(flat_cloth_middle_mask_np.transpose(2, 0, 1)).float()
        flat_cloth_right_mask_tensor = torch.tensor(flat_cloth_right_mask_np.transpose(2, 0, 1)).float()
        flat_cloth_label_tensor = torch.tensor(flat_cloth_label_np.transpose(2, 0, 1)).float()
        '''[1, 256, 192] => [1, 512, 384]'''

        WC_tensor = None    # 變形後衣服
        WE_tensor = None    # 衣物邊界
        AMC_tensor = None   # 手臂顏色
        ANL_tensor = None   # 手臂和頸部標籤
        warped_result_np = None

        # 合成網路再使用
        if self.warproot:
            '''skin color，提取皮膚顏色'''
            face_mask_np = (parsing_np==14).astype(np.uint8)    #用分割結果生成臉部遮罩
            skin_mask_np = (face_mask_np+hand_mask_np+neck_mask_np).astype(np.uint8)    #臉、手、頸部遮罩相加成整體皮膚遮罩
            P_np_resized = cv2.resize(P_np, (skin_mask_np.shape[1], skin_mask_np.shape[0])) # (修改)
            skin = skin_mask_np * P_np_resized  #將遮罩套用到原始圖片，只保留皮膚區域的rgb值

            # 將三維的rgb變成一維向量
            skin_r = skin[..., 0].reshape((-1))
            skin_g = skin[..., 1].reshape((-1))
            skin_b = skin[..., 2].reshape((-1))
            # 過濾出每個通道中非零值的索引，以便後續統計皮膚顏色的有效像素。
            skin_r_valid_index = np.where(skin_r > 0)[0]
            skin_g_valid_index = np.where(skin_g > 0)[0]
            skin_b_valid_index = np.where(skin_b > 0)[0]

            # 分別計算r, g, b中位數，當作統一顏色值
            skin_r_median = np.median(skin_r[skin_r_valid_index])
            skin_g_median = np.median( skin_g[skin_g_valid_index])
            skin_b_median = np.median(skin_b[skin_b_valid_index])

            # 用中位數算出手臂的通道顏色圖
            arms_r = np.ones_like(parsing_np[...,0:1]) * skin_r_median
            arms_g = np.ones_like(parsing_np[...,0:1]) * skin_g_median
            arms_b = np.ones_like(parsing_np[...,0:1]) * skin_b_median
            
            #將r, g, b合併，並轉換維度為(channel, height, width)
            arms_color = np.concatenate([arms_r,arms_g,arms_b],2).transpose(2,0,1)

            AMC_tensor = torch.FloatTensor(arms_color)  # 將np陣列轉成torch張量
            AMC_tensor = AMC_tensor / 127.5 - 1.0   #標準化[-1, 1]

            '''提取背景顏色'''
            # 假設 parsing_np 是分割圖像，0 代表背景
            background_mask_np = (parsing_np == 0).astype(np.uint8)    # 用分割結果生成背景遮罩
            P_np_resized_background = cv2.resize(P_np, (background_mask_np.shape[1], background_mask_np.shape[0])) # 修改圖像大小

            # 將遮罩應用到原始圖像，只保留背景區域的 RGB 值
            background = background_mask_np * P_np_resized_background

            # 將三維的 RGB 轉換為一維向量
            background_r = background[..., 0].reshape((-1))
            background_g = background[..., 1].reshape((-1))
            background_b = background[..., 2].reshape((-1))

            # 過濾出每個通道中非零值的索引，以便後續統計背景顏色的有效像素
            background_r_valid_index = np.where(background_r > 0)[0]
            background_g_valid_index = np.where(background_g > 0)[0]
            background_b_valid_index = np.where(background_b > 0)[0]

            # 分別計算 R, G, B 中位數或平均值，作為背景顏色
            background_r_median = np.median(background_r[background_r_valid_index])
            background_g_median = np.median(background_g[background_g_valid_index])
            background_b_median = np.median(background_b[background_b_valid_index])

            # 根據計算的顏色創建背景顏色張量
            background_r = np.ones_like(parsing_np[..., 0:1]) * background_r_median
            background_g = np.ones_like(parsing_np[..., 0:1]) * background_g_median
            background_b = np.ones_like(parsing_np[..., 0:1]) * background_b_median

            # 合併 R, G, B 通道並轉換為 (channel, height, width)
            background_color = np.concatenate([background_r, background_g, background_b], 2).transpose(2, 0, 1)

            # 將 numpy 陣列轉換為 torch 張量
            background_color_tensor = torch.FloatTensor(background_color)
            background_color_tensor = background_color_tensor / 127.5 - 1.0   # 標準化到 [-1, 1] 範圍

            '''warped clothes'''
            warped_name = person_id + '___' + cloth_id[:-4] +'.png'   # 00057_00.jpg___11274_00.png
            warped_path = os.path.join(self.warproot, warped_name)
            warped_result = Image.open(warped_path).convert('RGB')
            warped_result_np = np.array(warped_result)
            
            seg_path = os.path.join(self.segroot, warped_name)
            seg_result = Image.open(seg_path).convert('L')
            seg_result_np = np.array(seg_result)
            # print("warped_result_np.shape: ", np.unique(warped_result_np))
            # print("seg_result_np.shape: ", np.unique(seg_result_np))
            
            # warped_cloth_np = warped_result_np[:,-2*w:-w,:]
            # warped_parse_np = warped_result_np[:,-w:,:]

            warped_cloth = Image.fromarray(warped_result_np).convert('RGB')
            WC_tensor = transform_for_rgb(warped_cloth)

            warped_edge_np = (seg_result_np==5).astype(np.uint8) + \
                             (seg_result_np==21).astype(np.uint8) + \
                             (seg_result_np==22).astype(np.uint8)
            
            warped_edge = Image.fromarray(warped_edge_np).convert('L')
            WE_tensor = transform_for_mask(warped_edge) * 255.0
            WE_tensor = WE_tensor[0:1,...]
            
            arms_neck_label = (seg_result_np==11).astype(np.uint8) * 1 + \
                              (seg_result_np==15).astype(np.uint8) * 2 + \
                              (seg_result_np==16).astype(np.uint8) * 3

            arms_neck_label = Image.fromarray(arms_neck_label).convert('L')
            ANL_tensor = transform_for_mask(arms_neck_label) * 255.0 / 3.0
            ANL_tensor = ANL_tensor[0:1,...]
        
        
        input_dict = {
            'image': P_tensor, 
            'pose':Pose_tensor , 
            'densepose':dense_mask_tensor,
            'seg_gt': seg_gt_tensor, 
            'seg_gt_onehot': seg_gt_onehot_tensor,
            'person_clothes_mask': person_clothes_mask_tensor,
            'person_clothes_left_mask': person_clothes_left_sleeve_mask_tensor,
            'person_clothes_middle_mask': person_clothes_torso_mask_tensor,
            'person_clothes_right_mask': person_clothes_right_sleeve_mask_tensor,
            'preserve_mask': preserve_mask1_tensor, 
            'preserve_mask2': preserve_mask2_tensor,
            'preserve_mask3': preserve_mask3_tensor,
            'cloth': C_tensor, 'edge': CM_tensor, 
            'flat_clothes_left_mask': flat_cloth_left_mask_tensor,
            'flat_clothes_middle_mask': flat_cloth_middle_mask_tensor,
            'flat_clothes_right_mask': flat_cloth_right_mask_tensor,
            'flat_clothes_label': flat_cloth_label_tensor,
            'cloth_path': C_path,
            'img_path': P_path,
            'person_id': person_id, 
            'cloth_id': cloth_id, 
            'parsing_np': parsing_np
        }

        if WC_tensor is not None:
            input_dict['warped_cloth'] = WC_tensor
            input_dict['warped_edge'] = WE_tensor
            input_dict['arms_color'] = AMC_tensor
            input_dict['arms_neck_lable'] = ANL_tensor
            input_dict['background_color'] = background_color_tensor

        return input_dict

    def __len__(self):
        if self.mode == 'train':
            return len(self.P_paths) // (self.opt.batchSize * self.opt.num_gpus) * (self.opt.batchSize * self.opt.num_gpus)
        else:
            return len(self.P_paths)

    def name(self):
        return 'AlignedDataset'
