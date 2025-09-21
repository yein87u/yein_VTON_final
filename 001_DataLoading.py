
import os
import os.path as osp
from tqdm import tqdm
import json
import pickle

root = '/mnt/d/VITON-HD/VITON-HD1024_Origin'

def DataLoading(phase):
    datas = []
    with open(os.path.join("/mnt/d/VITON-HD/VITON-HD1024_Origin", f"{phase}_pairs.txt"), 'r') as f:
        for line in tqdm(f.readlines()):
            base_name = line.split()[0][:8] # 13:21
            test_cloth_name = line.split()[1][:8]

            images_dir = os.path.join(root, phase, 'image', f'{base_name}.jpg').replace("\\", "/")

            if(phase == 'train'):
                cloth_dir = os.path.join(root, phase, 'cloth', f'{base_name}.jpg').replace("\\", "/")
                cloth_mask_dir = os.path.join(root, phase, 'cloth-mask', f'{base_name}.jpg').replace("\\", "/")
                cloth_component_mask_dir = os.path.join('/mnt/d/VITON-HD/VITON-HD1024_Parsing', phase, 'cloth_parse-bytedance', f'{base_name}.png').replace("\\", "/")
            else:
                cloth_dir = os.path.join(root, phase, 'cloth', f'{test_cloth_name}.jpg').replace("\\", "/")
                cloth_mask_dir = os.path.join(root, phase, 'cloth-mask', f'{test_cloth_name}.jpg').replace("\\", "/")
                cloth_component_mask_dir = os.path.join('/mnt/d/VITON-HD/VITON-HD1024_Parsing', phase, 'cloth_parse-bytedance', f'{test_cloth_name}.png').replace("\\", "/")

            keypoints_dir = os.path.join(root, phase, 'openpose_json', f'{base_name}_keypoints.json').replace("\\", "/")
            image_densepose_dir = os.path.join(root, phase, 'image-densepose', f'{base_name}.jpg').replace("\\", "/")
            parse_dir = os.path.join('/mnt/d/VITON-HD/VITON-HD1024_Parsing', phase, 'parse-bytedance', f'{base_name}.png').replace("\\", "/")


            #檢查檔案是否存在
            if not os.path.isfile(images_dir):
                print(phase, 1)
                continue
            if not os.path.isfile(cloth_dir):
                print(phase, 2)
                continue
            if not os.path.isfile(cloth_mask_dir):
                print(phase, 3)
                continue
            if not os.path.isfile(cloth_component_mask_dir):
                print(phase, 4)
                continue
            if not os.path.isfile(keypoints_dir):
                print(phase, 5)
                continue
            if not os.path.isfile(image_densepose_dir):
                print(phase, 6)
                continue
            if not os.path.isfile(parse_dir):
                print(phase, 7)
                continue
            
            datas.append({
                'base_name': base_name,
                'test_cloth_name': test_cloth_name,
                'images_dir': images_dir,
                'cloth_dir': cloth_dir,
                'cloth_mask_dir': cloth_mask_dir,
                'cloth_component_mask_dir': cloth_component_mask_dir,
                'keypoints_dir': keypoints_dir,
                'image_densepose_dir': image_densepose_dir,
                'parse_dir': parse_dir
            })
    
    return datas



train_datas = DataLoading('train')
test_datas = DataLoading('test')

data_file = {'train':train_datas, 'test':test_datas}
pickle.dump(data_file, open('/mnt/c/Users/User/Desktop/yein_VTON/data/All_Data.pkl', 'wb')) #pickle序列化，二進位寫入檔案，wb為覆寫，ab為追加模式

print(train_datas[0])
print(test_datas[0])
print("train_datas_len: ", len(train_datas))





