import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

# 取得、切割圖片
def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize    # 寬高全都變loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w   # 高度根據寬高比例做等比縮放

    # 隨機生成裁切位置
    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    # flip 是控制圖片是否要左右反轉，進行資料增強的動作
    #flip = random.random() > 0.5
    flip = 0
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True, isTrain=True):
    # 將要做的transform動作存起來
    transform_list = []

    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]    # 調整成正方形
        transform_list.append(transforms.Resize(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))  #設定圖片寬度，依比例改圖片高度
        osize = [512, 384]
        transform_list.append(transforms.Resize(osize, method))  
    
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize))) # 進行圖片裁切

    # 若沒有給定參數，則將圖片調整成2的冪次方大小
    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    # 預處理的水平翻轉
    if isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    # 將 PIL 影像或 NumPy 陣列轉換為 PyTorch 張量
    transform_list += [transforms.ToTensor()]   # 影像維度從(height, width, channel) 轉換為 (channel, height, width)

    # 標準化
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    '''用+=代表transform_list = [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    '''
    return transforms.Compose(transform_list)


# 將圖像寬高調整成最接近base倍數的值，使用雙三次差值法
def __make_power_2(img, base, method=Image.BICUBIC):    #method可替換成 Image.LANCZOS，會產生較清晰的結果，但運行較慢
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

#調整寬度，並依比例調整高度，使用雙三次差值法
def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

# 將圖像裁剪成正方形，若圖片小於裁剪位置，則保持原樣
def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

#根據flip做圖片水平翻轉
def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img