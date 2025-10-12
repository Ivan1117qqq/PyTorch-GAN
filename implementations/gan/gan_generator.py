import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
import os
import argparse

from PIL import Image

# 參數設定（要與訓練時一致）

parser = argparse.ArgumentParser()
parser.add_argument("--n_images", type=int, default=100, help="要產生多少張圖片")
opt = parser.parse_args()

latent_dim = 100
img_size = 28
channels = 1
img_shape = (channels, img_size, img_size)

# 創建儲存資料夾
os.makedirs("generated", exist_ok=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

# 初始化模型
generator = Generator()

# 選擇裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)

# 載入訓練好的權重
generator.load_state_dict(torch.load("saved_model/generator.pth", map_location=device))
generator.eval()  # 關閉 BatchNorm 等訓練特性

# 產生 latent noise
num_samples = 25
z = torch.randn(num_samples, latent_dim).to(device)

# ========= 產生圖片 =========
z = torch.randn(opt.n_images, latent_dim).to(device)
gen_imgs = generator(z)

# ========= 儲存圖片 =========
for idx in range(opt.n_images):
    save_image(gen_imgs[idx], f"generated/img_{idx+1}.png", nrow=10, normalize=True)

print(f"成功產生 {opt.n_images} 張圖片，儲存於 generated/")


# ========= 儲存圖片 =========
for idx in range(opt.n_images):
    path = f"generated/img_{idx+1}.png"
    
    # 儲存圖片（原始）
    save_image(gen_imgs[idx], path, nrow=10, normalize=True)
    
    # 強制轉成灰階（避免圖片瀏覽器顯示成 RGB）
    img = Image.open(path).convert("L")
    img.save(path)  # 覆蓋原圖（或換成其他檔名以保留）

print(f"成功產生 {opt.n_images} 張圖片，儲存於 generated/")
