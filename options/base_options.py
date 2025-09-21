import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="超參數設置")
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='/mnt/d/VITON-HD')
        self.parser.add_argument('--warproot', type=str, default='')
        self.parser.add_argument('--segroot', type=str, default='')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='模型放在這裡')
        self.parser.add_argument('--checkpoint', type=str, default='')
        self.parser.add_argument('--image_all_data', type=str, default='./data/All_Data.pkl', help='圖像檔案位置')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='加載時縮放和裁剪圖像[resize_and_crop(調整大小和裁剪)|crop(裁剪)|scale_width(比例寬度)|scale_width_and_crop(縮放寬度和裁切)]')
        
        self.parser.add_argument('--name', type=str, default='flow', help='實驗名稱，儲存樣本與模型')   
        
        self.parser.add_argument('--num_gpus', type=int, default=1, help='GPU數量')
        self.parser.add_argument('--batchSize', type=int, default=2, help='批次大小')
        self.parser.add_argument('--image_size', type=int, default=512, help='模型在訓練和推理過程中處理的圖像大小')
        self.parser.add_argument('--loadSize', type=int, default=512, help='將影像縮放至此尺寸')
        self.parser.add_argument('--fineSize', type=int, default=512, help='圖片裁切至此尺寸')

        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='控制netG的全局下採樣層數量')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='控制netG的局部增強層的數量')

        self.parser.add_argument('--no_flip', action='store_true', help='若指定了此參數，則資料增強中不會做水平翻轉的處理')
        self.parser.add_argument('--verbose', action='store_true', default=False, help='自訂切換輸出詳細、簡潔模式')
        self.parser.add_argument('--previous_step', type=int, default=0)

        self.parser.add_argument('--predict', type=str, default='')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()

        # 解析命令行參數
        self.opt = self.parser.parse_args()
        self.previous_step = self.opt.previous_step
        
        if self.previous_step == 0:
            self.checkpoint = ''
        else:
            self.checkpoint = os.path.join(self.checkpoints_dir, self.name, 'step_%06d.pth' % (self.previous_step))


        print(self.opt.name)

        return self.opt