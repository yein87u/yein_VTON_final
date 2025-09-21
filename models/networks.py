import os
import torch
import functools
import torch.nn as nn
from options import base_options
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint
from torchvision import models

opt = base_options.BaseOptions().parse()

def save_checkpoint(model, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    model.to(device)

# 讓模型在訓練或推理時使用之前訓練過的參數
def load_checkpoint_parallel(model, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return

    checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(0))
    checkpoint_new = model.state_dict()
    # 遍歷整個參數名稱，每個參數皆包含一組權重
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]
    # 更新參數
    model.load_state_dict(checkpoint_new)
    model.to(device)

# 控制模型參數是否參與梯度計算，訓練判別器與生成器時的切換
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]   # 單一網路轉成列表

    # 逐一設定網絡參數的 requires_grad
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

'''
定義使用LSGAN或常規GAN的GAN損失。
使用LSGAN時, 與MSELoss基本相同, 但它抽象化了創建目標標籤張量的需要與輸入的大小相同
'''
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        # 將標籤註冊為模型的緩存張量，不會跟著模型梯度更新
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        # 決定使用哪種誤差損失
        if use_lsgan:
            self.loss = nn.MSELoss()    # LSGAN損失(均方誤差損失)
        else:
            self.loss = nn.BCELoss()    # 二元交叉熵損失
        
        # 確保模型損失是三個其中的一個
        assert gan_mode in ['lsgan', 'vanilla', 'wgangp']

        # 若使用的是 WGAN-GP，這是基於 Wasserstein 距離的損失，通常會涉及梯度懲罰
        if gan_mode in ['wgangp']:
            self.loss = None    # 因此不需要計算標準的LOSS
        self.gan_mode = gan_mode
        '''
        Wasserstein距離衡量的是將一個概率分佈"轉換"為另一個概率分佈所需要的最小工作量。
        這個“工作量”可以視為從一個分佈到另一個分佈的“移動”成本, 這樣的度量通常用於計算不同分佈之間的相似度或距離。
        WGAN引入了這種損失來改善傳統GAN的訓練穩定性, 也就是WGAN-GP進一步通過梯度懲罰來增強訓練的穩定性
        '''

    def get_target_tensor(self, input, target_is_real):
        
        # 將目標標籤設為相對應的標籤
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        
        # 將目標標籤擴展到與input相同的形狀，使GAN損失函數將模型預測（prediction）和目標標籤進行比較
        return target_tensor.expand_as(input)

    def __call__(self, prediction, target_is_real, add_gradient=False):
        if self.gan_mode in ['lsgan', 'vanilla']:
            # 取得真實標籤
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # 取得 loss 值
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()  # + 0.001*(prediction**2).mean()
                if add_gradient:
                    loss = -prediction.mean() + 0.001*(prediction**2).mean()
            else:
                loss = prediction.mean()
        return loss

# VGG19 預訓練模型
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # 載入 VGG19 預訓練模型，並提取卷積層，非全連接層
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        # 按照提取出來的卷積層分割成5個子網路
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # 將模型參數設為不可訓練，使得反向傳播時權重不會更新 
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # 讓圖像依次通過子網路
        '''if size: 256*192
        X:  torch.Size([12, 64, 16, 12])
        X1:  torch.Size([12, 64, 16, 12])
        X2:  torch.Size([12, 128, 8, 6])
        X3:  torch.Size([12, 256, 4, 3])
        X4:  torch.Size([12, 512, 2, 1])
        X5:  torch.Size([12, 512, 1, 0])
        '''
        # print("X: ", X.shape)
        # h_relu1 = checkpoint(self.slice1, X, use_reentrant=False)
        # h_relu2 = checkpoint(self.slice2, h_relu1, use_reentrant=False)
        # h_relu3 = checkpoint(self.slice3, h_relu2, use_reentrant=False)
        # h_relu4 = checkpoint(self.slice4, h_relu3, use_reentrant=False)
        # h_relu5 = checkpoint(self.slice5, h_relu4, use_reentrant=False)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        # 每個子網路都會產出中間特徵圖，多用於風格轉換或特徵匹配
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

# VGG19模型損失
class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()  # 初始化 VGG19 預訓練模型
        self.vgg.cuda() # 移動到 GPU 上進行
        self.criterion = nn.L1Loss()    # 使用 L1 損失作為衡量圖像差異的標準
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  # 代表不同子網路的特徵權重
        self.layids = layids    # 計算損失時選擇的層

    def forward(self, x, y):
        # x, y代表的是影像
        '''
        [12, 3, 16, 12]
        [12, 3, 32, 24]
        [12, 3, 64, 48]
        [12, 3, 128, 96]
        [12, 3, 256, 192]
        '''
        # print("x: ", x.shape)
        # print("y: ", y.shape)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y) # [12, 3, height, width], [12, 3, height, width]
        loss = 0
        # 若沒有指定是哪一層，則默認指定所有層
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        
        # 計算兩張影像的L1損失
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        # 回傳總損失為VGG損失
        return loss

# 基於 PatchGAN 架構的判別器，多用於對抗生成網路(GAN)
class SpectralDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, opt, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(SpectralDiscriminator, self).__init__()
        # 判斷是否要有bias，原因是 BatchNorm2d 自帶bias
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4  # 卷積核大小
        padw = 1    # 填充大小，確保卷積操作的空間特徵不會太快縮小

        # 將特徵維度縮小，並提升通道數，使之後的卷積能夠捕抓到更多更複雜的特徵
        # 譜歸一化避免梯度消失、爆炸現象發生，LeakyReLU 使網絡能夠學習到更複雜的模式
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # 逐漸增加 filters 數量，也就是輸出數量
            nf_mult_prev = nf_mult
            # 每個輸出對應一個濾波器，因此濾波器是隨著指數成長，最多到 8 倍
            nf_mult = min(2 ** n, 8)    # n: 1, 2   nf_mult = 2, 4
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)), nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        
        # 檢查是否使用 Sigmoid 函數使輸出控制在[0, 1]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        
        # 將 sequence 一層一層傳入，讓模型按照順序執行，使用在層數較多的模型上
        self.model = nn.Sequential(*sequence)
        self.old_lr = opt.lr_D

    def forward(self, input):
        """Standard forward."""
        return checkpoint(self.model, input, use_reentrant=False)
    
    def update_learning_rate(self, optimizer, opt):
        lrd = opt.lr_D / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.local_rank == 0:
            print('update learning rate for D model: %f -> %f' %(self.old_lr, lr))
        self.old_lr = lr


'''tryon network'''
class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, opt=None):
        super(ResUnetGenerator, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        # 設定 old_lr
        if opt is None:
            print("opt 參數未提供！")
            raise ValueError("opt 參數未提供！")

        self.model = unet_block
        self.old_lr = opt.lr
        self.old_lr_gmm = 0.1*opt.lr

    def forward(self, input):
        return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)



class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out