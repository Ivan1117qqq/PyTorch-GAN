import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
import os

import argparse

from PIL import Image

# 同樣的Generator架構定義 (要跟訓練時一樣)
class Generator(nn.Module):
    def __init__(self,latent_dim,img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

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
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

def generate_images():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=100, help="要產生多少張圖片")
    parser.add_argument("--latent_dim", type=int, default=100, help="潛在向量維度")
    parser.add_argument("--img_size", type=int, default=32, help="圖片尺寸")
    parser.add_argument("--channels", type=int, default=1, help="圖片通道數")
    parser.add_argument("--model_path", type=str, default="saved_model/generator.pth", help="Generator 模型路徑")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="生成圖片儲存資料夾")
    opt = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成器初始化
    img_shape = (1, 28, 28)
    generator = Generator(latent_dim=opt.latent_dim, img_shape=img_shape).to(device)
    generator.load_state_dict(torch.load("saved_model/generator.pth", map_location=device))
    generator.eval()

    # 產生latent noise
    z = torch.randn(opt.n_images, opt.latent_dim).to(device)

    # 產生圖片
    gen_imgs = generator(z)

    # 建立資料夾存圖
    os.makedirs("generated_images", exist_ok=True)
    for i in range(opt.n_images):
        save_image(gen_imgs[i], f"generated_images/img_{i+1}.png", normalize=True)

        path = os.path.join(opt.output_dir, f"img_{i+1}.png")
        # 讀回並強制轉為灰階再儲存（確保通道為 1）
        img = Image.open(path).convert("L")
        img.save(path)  # 覆蓋原圖（也可改名保留原始圖）

    print(f"{opt.n_images}張圖片已生成並儲存在 generated_images 資料夾")



if __name__ == "__main__":
    generate_images()
