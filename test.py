# def get_tuple_shapes(my_tuple):
#     return tuple(
#         element.shape if hasattr(element, 'shape') else f"<{type(element).__name__}>"
#         for element in my_tuple
#     )


import torch
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

# 檢查 GPU 資訊
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
# import triton
# print("triton.__version__: ", triton.__version__)


# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
# import cv2 as cv
# import numpy as np
# import argparse

# from options.train_options import TrainOptions
# opt = TrainOptions().parse()
# # parser = argparse.ArgumentParser()
# # parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
# # parser.add_argument('--width', default=512, type=int, help='Resize input to specific width.')
# # parser.add_argument('--height', default=384, type=int, help='Resize input to specific height.')

# # args = parser.parse_args()

# BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

# POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# inWidth = 384
# inHeight = 512

# net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# cap = cv.VideoCapture('/mnt/d/VITON-HD/VITON-HD1024_Origin/train/image/00132_00.jpg')

# while cv.waitKey(1) < 0:
#     hasFrame, frame = cap.read()
#     if not hasFrame:
#         cv.waitKey()
#         break

#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]
    
#     net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
#     out = net.forward()
#     out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

#     assert(len(BODY_PARTS) == out.shape[1])

#     points = []
#     for i in range(len(BODY_PARTS)):
#         # Slice heatmap of corresponging body's part.
#         heatMap = out[0, i, :, :]

#         # Originally, we try to find all the local maximums. To simplify a sample
#         # we just find a global one. However only a single pose at the same time
#         # could be detected this way.
#         _, conf, _, point = cv.minMaxLoc(heatMap)
#         x = (frameWidth * point[0]) / out.shape[3]
#         y = (frameHeight * point[1]) / out.shape[2]
#         # Add a point if it's confidence is higher than threshold.
#         points.append((int(x), int(y)) if conf > 0.2 else None)

#     for pair in POSE_PAIRS:
#         partFrom = pair[0]
#         partTo = pair[1]
#         assert(partFrom in BODY_PARTS)
#         assert(partTo in BODY_PARTS)

#         idFrom = BODY_PARTS[partFrom]
#         idTo = BODY_PARTS[partTo]

#         if points[idFrom] and points[idTo]:
#             cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
#             cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
#             cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

#     t, _ = net.getPerfProfile()
#     freq = cv.getTickFrequency() / 1000
#     cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

#     cv.imwrite("/mnt/c/Users/User/Desktop/yein_VTON/sample/checkdata/keypointcheck/00132_00.jpg", frame)  # 儲存代替顯示


'''顯示圖片'''
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# # img_path = 'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\ptexttp\\test\\03921_00.jpg___08015_00.png'
# img_path = 'D:\\VITON-HD\\VITON-HD1024_Origin\\test\\image-densepose\\00009_00.png'
# img = mpimg.imread(img_path)


# plt.imshow(img)
# plt.axis('off')  # 隱藏坐標軸
# plt.show()

'''確認最大、最小數字'''
# import os
# from PIL import Image
# import numpy as np
# from tqdm import tqdm

# def get_files(path):
#     files = os.listdir(path)
#     return files

# files = get_files('D:\\VITON-HD\\VITON-HD1024_Parsing\\test\\parse-bytedance')

# max_value = float('-inf')
# min_value = float('inf')
# for path in tqdm(files, desc="Processing images"):
#     # print(path)
#     img = Image.open(os.path.join('D:\\VITON-HD\\VITON-HD1024_Parsing\\test\\parse-bytedance', path))
#     image_gray = img.convert('L')
#     image_gray = np.array(image_gray)
#     x, y = image_gray.shape[0], image_gray.shape[1]
#     for i in range(x):
#         for j in range(y):
#             if(image_gray[i][j] < min_value):
#                 min_value = image_gray[i][j]
#             if(image_gray[i][j] > max_value):
#                 max_value = image_gray[i][j]

#     # pixels = list(image_gray.getdata())
#     # max_temp = max(pixels)
#     # min_temp = min(pixels)


#     # if(max_temp > max_value):
#     #     max_value = max_temp
#     # if(min_temp < min_value):
#     #     min_value = min_temp

#     # print('max: ', max_value, 'min: ', min_value)
#     # plt.imshow(image_gray)
#     # plt.axis('off')  # 隱藏坐標軸
#     # plt.show()
#     # break
# print('max: ', max_value, 'min: ', min_value)

'''參數量'''
# from models import AFWM
# from options.train_options import TrainOptions

# opt = TrainOptions().parse()

# warp_model = AFWM.AFWM_Vitonhd_lrarms(opt, 51)

# # total = sum([param.nelement() for param in warp_model.parameters()])
# # print('Number of parameter: %.2fM' % (total/1e6))


'''計算量'''
# import torch
# from thop import profile
# from models import AFWM
# from options.train_options import TrainOptions
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

# opt = TrainOptions().parse()
# warp_model = AFWM.AFWM_Vitonhd_lrarms(opt, 51)
# warp_model.train()
# warp_model.cuda()
# device = torch.device(f'cuda:{opt.local_rank}')
# model = warp_model.to(device)

# concat = torch.randn(4, 51, 512, 384)
# clothes = torch.randn(4, 3, 512, 384)
# pre_clothes_edge = torch.randn(4, 1, 512, 384)
# cloth_parse_for_d = torch.randn(4, 1, 512, 384)
# clothes_left = torch.randn(4, 3, 512, 384)
# clothes_torso = torch.randn(4, 3, 512, 384)
# clothes_right = torch.randn(4, 3, 512, 384)
# left_cloth_sleeve_mask = torch.randn(4, 1, 512, 384)
# cloth_torso_mask = torch.randn(4, 1, 512, 384)
# right_cloth_sleeve_mask = torch.randn(4, 1, 512, 384)
# preserve_mask3 = torch.randn(4, 1, 512, 384)

# input = (concat, clothes, pre_clothes_edge, cloth_parse_for_d, clothes_left, 
#           clothes_torso, clothes_right, left_cloth_sleeve_mask, cloth_torso_mask, 
#           right_cloth_sleeve_mask, preserve_mask3)
# input = [tensor.to(device) for tensor in input]  # 如果 input 是張量列表

# flops, params = profile(model, inputs=(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8], input[9], input[10]))
# flops = FlopCountAnalysis(model, input).total()
# params = parameter_count_table(model)

# print('flops: ', flops/1e9, 'params: ', params/1e6)
# print('params: ', params)


# from models.networks import AttU_Net

# model = AttU_Net(38, 3)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"AttU_Net 的可訓練參數量: {total_params}")


# import torch
# from fvcore.nn import FlopCountAnalysis, flop_count_table

# from models.LightMUNet import LightMUNet
# from options.train_options import TrainOptions
# from models.networks import ResUnetGenerator
# from models.CMUnet import VSSM
# import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())

# def print_model_info(model, input_tensor, name="Model"):
#     model.eval()
#     model.to(device)
#     input_tensor = input_tensor.to(device)

#     # FLOPs
#     flops = FlopCountAnalysis(model, input_tensor)
#     print(f"\n--------- {name} ---------")
#     print(flop_count_table(flops, max_depth=2))  # 控制顯示層級

#     # Params
#     total_params = count_parameters(model)
#     print(f"Total Parameters: {total_params:,}")

# Dummy input
# dummy_input = torch.randn(2, 36, 512, 384).to(device)  # BCHW

# # LightMUNet
# model1 = LightMUNet(spatial_dims=2, init_filters=8, in_channels=36, out_channels=4)
# print_model_info(model1, dummy_input, name="LightMUNet")

# # ResUnetGenerator
# opt = TrainOptions().parse()
# opt.warproot = 'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\result\\train'
# opt.segroot = 'c:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\seg\\train'
# model2 = ResUnetGenerator(36, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d, opt=opt)
# print_model_info(model2, dummy_input, name="ResUnetGenerator")

# # VSSM (CNN-Mamba-Unet)
# model3 = VSSM()
# print_model_info(model3, dummy_input, name="CNN-Mamba-UNet")

# from ptflops import get_model_complexity_info
# import torch
# from models import AFWM

# class Opt:
#     def __init__(self):
#         self.lr = 0.0001

# opt = Opt()
# model = AFWM.AFWM_Vitonhd_lrarms(opt, input_nc=51).cuda()
# model.eval()

# def input_constructor(input_res):
#     return {
#         'cond_input': torch.randn(2, 51, 512, 384).cuda(),
#         'image_input': torch.randn(2, 3, 512, 384).cuda(),
#         'image_edge': torch.randn(2, 1, 512, 384).cuda(),
#         'image_label_input': torch.randn(2, 1, 512, 384).cuda(),
#         'image_input_left': torch.randn(2, 3, 512, 384).cuda(),
#         'image_input_torso': torch.randn(2, 3, 512, 384).cuda(),
#         'image_input_right': torch.randn(2, 3, 512, 384).cuda(),
#         'image_edge_left': torch.randn(2, 1, 512, 384).cuda(),
#         'image_edge_torso': torch.randn(2, 1, 512, 384).cuda(),
#         'image_edge_right': torch.randn(2, 1, 512, 384).cuda(),
#         'preserve_mask': torch.randn(2, 1, 512, 384).cuda()
#     }

# macs, params = get_model_complexity_info(
#     model,
#     input_res=(2, 51, 512, 384),  # 跟 cond_input 對應
#     input_constructor=input_constructor,
#     as_strings=True,
#     print_per_layer_stat=False,
#     verbose=False
# )

# print(f"FLOPs: {macs}")
# print(f"Params: {params}")




