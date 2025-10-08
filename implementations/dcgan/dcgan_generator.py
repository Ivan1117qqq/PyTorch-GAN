import argparse
import os
import numpy as np
import torch
from torchvision.utils import save_image

import torch.nn as nn

# 定義 Generator 跟訓練時相同
class Generator(nn.Module):
    def __init__(self,img_size,latent_dim,channels):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=25, help="要產生多少張圖片")
    parser.add_argument("--latent_dim", type=int, default=100, help="潛在向量維度")
    parser.add_argument("--img_size", type=int, default=32, help="圖片尺寸")
    parser.add_argument("--channels", type=int, default=1, help="圖片通道數")
    parser.add_argument("--model_path", type=str, default="saved_model/generator.pth", help="Generator 模型路徑")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="生成圖片儲存資料夾")
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立 generator 並載入權重
    generator = Generator(img_size = opt.img_size,latent_dim = opt.latent_dim, channels = opt.channels).to(device)
    generator.load_state_dict(torch.load(opt.model_path, map_location=device))
    generator.eval()

    # 產生 latent noise
    z = torch.randn(opt.n_images, opt.latent_dim).to(device)

    # 產生圖片
    with torch.no_grad():
        gen_imgs = generator(z)

    # 存圖片，每張圖片一個檔案
    for i in range(opt.n_images):
        save_image(gen_imgs[i], os.path.join(opt.output_dir, f"img_{i+1}.png"), normalize=True)

    print(f"✅ 成功產生 {opt.n_images} 張圖片，存放在 {opt.output_dir}/")

if __name__ == "__main__":
    main()
