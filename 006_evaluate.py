import os
import torch
import numpy as np
import cv2
import torch_fidelity
from lpips import LPIPS
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn import functional as F
from math import exp


from options.train_options import TrainOptions
from data import aligned_dataset
from torch.utils.data import DataLoader


def CreateDataset(opt):
    dataset = aligned_dataset.AlignedDataset()
    dataset.initialize(opt, mode="test")
    
    return dataset

opt = TrainOptions().parse()
opt.evaluate = 'true'

run_path = 'runs/train_tryon/'+opt.name         # 'runs/flow'
sample_path = 'sample/train_tryon/'+opt.name    # 'sample/flow'
os.makedirs(run_path, exist_ok=True)
os.makedirs(sample_path, exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

evaluate_data = CreateDataset(opt)     # 11632 不是 11647是因為batchsize分割時data沒辦法被整除
evaluate_loader = DataLoader(evaluate_data, batch_size=opt.batchSize, shuffle=True, num_workers=0, pin_memory=True)

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取影像並轉為 Tensor
def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):  # 確保排序一致
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)  # 讀取 BGR 圖片
        img = cv2.resize(img, (512, 384))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換為 RGB
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0  # 轉為 (C, H, W)
        images.append(img)
    return torch.stack(images) if images else None  # 回傳 (N, C, H, W) Tensor


def ssim_fn(img1, img2, window_size = 11, size_average = True):
    
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(img1, img2, window, window_size, channel, size_average = True):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    # assuming its a Dataset if not a Tensor
    if not isinstance(img1, torch.Tensor):
        img1 = torch.stack([s["image"]["I"] for s in iter(img1)], dim=0)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.stack([s["image"]["I"] for s in iter(img2)], dim=0)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def evaluate_metrics():
    '''
    GP_ResUNet
    LMUnet
    CMUnet
    '''
    model = 'GP_ResUNet'
    # 影像資料夾
    fake_images_folder = "/mnt/c/Users/User/Desktop/yein_VTON/sample/test_tryon/fake_images2/" + model + "/"
    real_images_folder = "/mnt/c/Users/User/Desktop/yein_VTON/sample/test_tryon/real_images2/GP_ResUNet/"
        
    os.makedirs(real_images_folder, exist_ok=True)
    
    # for i, data in enumerate(tqdm(evaluate_loader, desc="Evaluate", ncols=100)):
    #     for bb in range(2):
    #         real_image = data['image'][bb]
    #         person_id = data['person_id'][bb]
    #         cv_img = (real_image.permute(1, 2, 0).detach().cpu().numpy()+1)/2
    #         rgb = (cv_img*255).astype(np.uint8)
    #         bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(real_images_folder + person_id, bgr)


    # 讀取影像
    print("讀取fake影像中")
    fake_images = load_images_from_folder(fake_images_folder).to(device)
    print("讀取real影像中")
    real_images = load_images_from_folder(real_images_folder).to(device)

    if fake_images is None or real_images is None:
        print("無法載入影像，請確認資料夾是否正確！")
        exit()

    # 初始化 LPIPS
    lpips_fn = LPIPS(net='vgg', verbose=False).to(device)

    # 計算 LPIPS、SSIM
    lpips_scores = []
    ssim_scores = []

    with torch.no_grad():
        for i in tqdm(range(fake_images.shape[0]), desc="計算 LPIPS & SSIM"):
            fake_img = fake_images[i].unsqueeze(0).to(device)
            real_img = real_images[i].unsqueeze(0).to(device)
            
            lpips_score = lpips_fn(fake_img, real_img).item()
            lpips_scores.append(lpips_score)
            
            ssim_value = ssim_fn(fake_img, real_img, size_average=True).item()
            ssim_scores.append(ssim_value)

    # 計算 FID
    print("\n計算 FID...")
    metrics = torch_fidelity.calculate_metrics(
        input1=fake_images_folder,
        input2=real_images_folder,
        fid=True,
        batch_size=2,  # 可以先從 4 試，還是爆就試 2
        dataloader_kwargs=dict(pin_memory=False),
        cuda=True  # 確保是使用 GPU
    )

    fid_score = metrics['frechet_inception_distance']

    # 顯示結果
    print("\n**評估結果**")
    print(f"LPIPS: {np.mean(lpips_scores):.3f} ± {np.std(lpips_scores):.3f}")
    print(f"SSIM:  {np.mean(ssim_scores):.3f} ± {np.std(ssim_scores):.3f}")
    print(f"FID:   {fid_score:.3f}")

if __name__ == "__main__":
    evaluate_metrics()