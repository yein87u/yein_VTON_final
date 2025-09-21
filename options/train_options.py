from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--PBAFN_human_parsing_checkpoint', type=str, help='從指定位置載入預訓練模型')
        
        self.parser.add_argument('--PBAFN_warp_checkpoint', type=str, help='從指定位置讀取預訓練位置')
        self.parser.add_argument('--pretrain_checkpoint_D', type=str, help='從指定位置載入預訓練模型')
        self.parser.add_argument('--PBAFN_gen_checkpoint', type=str, help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_gen_LMUnet_checkpoint', type=str, help='load the pretrained model from the specified location')

        self.parser.add_argument('--local_rank', type=int, default=0, help='進行分布式訓練時的CPU編號')

        self.parser.add_argument('--niter', type=int, default=30, help='起始學習率的 iter 數量')
        self.parser.add_argument('--niter_decay', type=int, default=50, help='學習率從初始值逐漸減小到某個較小值所需的總訓練迭代數。')
        self.parser.add_argument('--lr', type=float, default=0.00005, help='Adam的初始學習率')
        self.parser.add_argument('--lr_D', type=float, default=0.00005, help='Adam的初始學習率')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='動量參數(控制一階動量估計)')

        self.parser.add_argument('--mask_epoch', type=int, default=-1)

        self.parser.add_argument('--write_loss_frep', type=int, default=100, help='在螢幕上顯示訓練結果的頻率')

        self.parser.add_argument('--first_order_smooth_weight', type=float, default=0.01)
        self.parser.add_argument('--display_freq', type=int, default=100, help='顯示訓練結果的頻率')

        self.parser.add_argument('--print_freq', type=int, default=11647, help='frequency of showing training results on console')  # 100
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')

        self.parser.add_argument('--resolution', type=int, default=512)
        self.parser.add_argument('--dataset', type=str, default='vitonhd')
        self.parser.add_argument('--no_dynamic_mask', action='store_true')

        self.parser.add_argument('--keep_step', type=int, default=3000)
        self.parser.add_argument('--decay_step', type=int, default=3000)

        self.isTrain = True