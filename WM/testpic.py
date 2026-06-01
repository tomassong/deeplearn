import os

import cv2
import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from datetime import datetime
import warnings
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ==================== 你的模块导入 ====================
from network import noise
from network.encoderdiff2 import Encoder
from network.decoderdiff2 import Decoder
from network.DiffFeatureExtractor import DifferenceLearner
from config import training_config as cfg
from utils import DTCWT_highpass
from utils.torch_utils import decoded_message_error_rate_batch
from network.noise import stargan_for_test
# =====================================================

# ==================== 配置 ====================
IMAGE_FOLDER = r"E:\scpcode\CelebAMask-HQ\processed\test33_128"
SAVE_ROOT = f"exp_15pics_15noises/{datetime.now().strftime('%Y.%m.%d-%H.%M.%S')}"
os.makedirs(SAVE_ROOT, exist_ok=True)

# 图片 → [-1, 1] Tensor
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

identity = noise.Identity()
#jpeg = noise.JpegCompression()
resize = noise.Resize()
medianblur = noise.MedianBlur()
gau_noise = noise.GaussianNoise()
gau_blur = noise.GaussianBlur()
dropout_noise = noise.Dropout()
salt_pepper_noise = noise.SaltPepper()
brightness_noise = noise.Brightness()
contrast_noise = noise.Contrast()
saturation_noise = noise.Saturation()
hue_noise = noise.Hue()
stargan = noise.stargan_noise
ganimation = noise.ganimation_noise
simswap = noise.simswap_noise
# 15 种扰动 + 对应类型（严格顺序！）
NOISE_LIST = [
    #("jpeg",         lambda inp, t: jpeg(inp),          "default"),
    ("resize",       lambda inp, t: resize(inp),                         "default"),
    ("medianblur",   lambda inp, t: medianblur(inp),                     "default"),
    ("gaublur",      lambda inp, t: gau_blur(inp),                    "default"),
    ("gaunoise",     lambda inp, t: gau_noise(inp),                  "default"),
    ("dropout",      lambda inp, t: dropout_noise(inp),                        "default"),
    ("saltpepper",   lambda inp, t: salt_pepper_noise(inp),                     "default"),
    ("identity",     lambda inp, t: identity(inp),                       "default"),
    ("brightness",   lambda inp, t: brightness_noise(inp),                     "default"),
    ("contrast",     lambda inp, t: contrast_noise(inp),                       "default"),
    ("saturation",   lambda inp, t: saturation_noise(inp),                     "default"),
    ("hue",          lambda inp, t: hue_noise(inp),                             "default"),
    ("simswap",      lambda inp, t: simswap(inp, "all"),     "all"),
    ("ganimation",   lambda inp, t: ganimation(inp, "all"),   "all"),
    ("stargan",      lambda inp, t: stargan(inp, "all"), "all"),
]

indices_encoder = torch.tensor([0, 1, 2]).to(cfg.device)

def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess(images):
    images_Y = images[:, [0], :, :]
    images_U = images[:, [1], :, :]
    images_V = images[:, [2], :, :]
    low_pass, high_pass = DTCWT_highpass.images_U_dtcwt_with_low(images_V)
    return images_Y, images_U, images_V, low_pass, high_pass

face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
def get_face_mask(image):
    """
    image: numpy RGB [H,W,3]
    return: torch [1,H,W]
    """

    H, W = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    mask = torch.zeros((1, H, W), dtype=torch.float32)

    for (x, y, w, h) in faces:
        mask[:, y:y+h, x:x+w] = 1.0

    # ⭐ 扩展（避免漏边界）
    mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=31, stride=1, padding=15).squeeze(0)

    # ⭐ 平滑（非常关键）
    mask = F.avg_pool2d(mask.unsqueeze(0), kernel_size=31, stride=1, padding=15).squeeze(0)

    return mask
# ==================== 仅定义 Encoder 的模型 ====================
class EncoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder().to(cfg.device)


def calculate_psnr_torch_neg1_1(img1, img2):
    """
    直接在 [-1, 1] 范围内用 PyTorch 计算 PSNR
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    # data_range = 2 因为 [-1, 1] 的范围是 2
    max_val = 2.0
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.item()
# ==================== 主程序 ====================
if __name__ == "__main__":
    seed_torch(42)

    # 1. 加载模型（只加载 encoder 部分）
    model = EncoderOnly()
    checkpoint = torch.load("E:/scpcode/Watermarking18/exp_highpass/2025.11.08-08.41.13/model_state_7.pth", map_location="cuda:0")
    encoder_state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_state_dict)
    model.encoder.eval()
    print("Encoder 加载成功！")

    # 2. 读取 15 张图片（按文件名排序）
    image_paths = sorted([
        os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    #assert len(image_paths) == 15, f"必须正好15张图，实际有 {len(image_paths)} 张"

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            noise_name, noise_fn, noise_type = NOISE_LIST[idx]

            # --- 读取图片 ---
            pil_img = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_img)

            img_tensor = to_tensor(pil_img).unsqueeze(0).to(cfg.device)  # [1,3,H,W] in [0,1]
            face_mask = get_face_mask(img_np)
            face_mask = face_mask.to(cfg.device)
            face_mask_unsqueezed = face_mask.unsqueeze(0)
            print('11111111111111', face_mask_unsqueezed.shape)
            img_tensor = img_tensor * 2.0 - 1.0  # → [-1,1]

            # --- 嵌入水印 ---
            Y, U, V, low_pass, high_pass = preprocess(img_tensor)


            #high_pass_extract = DTCWT_highpass.images_U_dtcwt_without_low(cover_images_V)
            # print("low_pass shape:", low_pass.shape)
            # print("low_pass shape11:", high_pass[1].shape)


            watermark = torch.Tensor(np.random.choice([-cfg.message_range, cfg.message_range],
                                                      (img_tensor.shape[0], cfg.message_length))).to(
                cfg.device)
            # selected_areas_embed = torch.index_select(high_pass[1], 2, indices_encoder)
            # selected_areas_embed = selected_areas_embed[:, :, :, :, :, 0].squeeze(1)
            selected = high_pass[1][:, :, indices_encoder, :, :, :]  # [B,1,3,H,W,2]

            # 提取实部、虚部，计算幅值和相位
            selected_areas_embed = selected[..., 0].squeeze(1)  # [B,1,3,H,W]
            # 验证阶段Encoder总是使用eval模式
            ans = model.encoder(selected_areas_embed, watermark, 0.08)
            ans = ans.unsqueeze(1)
            high_pass[1][:, :, indices_encoder, :, :, 0] = ans
            v_embedded = DTCWT_highpass.dtcwt_images_U(low_pass, high_pass)
            #v_clamped = torch.clamp(v_embedded, -1, 1)

            #v_embedded = v_embedded + (v_clamped - v_embedded).detach()
            watermarked_images = torch.cat([Y, U, v_embedded], dim=1)
            # 使用
            psnr_value = calculate_psnr_torch_neg1_1(img_tensor.squeeze(0), watermarked_images)
            print(f"PSNR: {psnr_value:.2f} dB")

            # --- 构造 input list（和训练一致）---
            forward_u_embedded = v_embedded.clone()
            forward_watermarked = watermarked_images.clone()
            forward_cover = img_tensor.clone()
            forward_mask = torch.zeros_like(img_tensor[:, :1, :, :])
            input_list = [V, forward_cover, forward_cover, forward_mask]

            # --- 应用扰动 ---
            if noise_name == "stargan":
                attacked, noised = noise_fn(input_list, noise_type)
            elif noise_name == "simswap":
                attacked, noised = noise_fn(input_list, noise_type)
            elif noise_name == "ganimation":
                attacked, noised = noise_fn(input_list, noise_type)
            else:
                attacked, noised = noise_fn(input_list, noise_type)
            attacked = attacked.clamp(-1, 1)

            # --- 保存路径 ---
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_dir = os.path.join(SAVE_ROOT, f"{idx:02d}_{noise_name}_{img_name}")
            os.makedirs(save_dir, exist_ok=True)

            # --- 保存函数 ---
            def save_tensor(tensor, path):
                save_image((tensor + 1) / 2, path)  # [-1,1] → [0,1]

            # 1. 原始图
            save_tensor(img_tensor, os.path.join(save_dir, "01_cover.png"))
            # 2. 含水印图
            watermarked_images = watermarked_images.squeeze(0)
            save_tensor(watermarked_images, os.path.join(save_dir, "02_watermarked.png"))
            # ==================== 高级可视化函数（已修复）===================
            # 3. 水印残差（×20）
            res_wm = (v_embedded - V).abs()   # 60~80 之间最漂亮
            save_tensor(res_wm, os.path.join(save_dir, "03_residual_wm.png"))
            # 4. 扰动后图
            save_tensor(noised+V, os.path.join(save_dir, "04_attacked.png"))
            # 5. 扰动残差（×20）
            res_noise = (attacked - watermarked_images).abs() * 1000  # 更准确：和含水印图比
            save_tensor(res_noise, os.path.join(save_dir, "05_attacked.png"))

            #res_noise = (v_embedded - V).abs() * 100  # 更准确：和含水印图比
            save_tensor(noised, os.path.join(save_dir, "06_attacked.png"))
            # residual_visualize(attacked, watermarked,
            #                    os.path.join(save_dir, "05_residual_noise.png"),
            #                    strength=2.0)
            diff = (v_embedded - V).abs().mean(dim=1)

            plt.imshow(diff[0].cpu(), cmap='hot')
            plt.savefig("diff.png")
            print(f"[{idx+1:02d}/15] {img_name} → 扰动: {noise_name} → 已保存")

    print(f"\n所有 15 张图处理完成！")
    print(f"结果保存在：{SAVE_ROOT}")
    print("每个文件夹包含：")
    print("  01_cover.png           → 原始图")
    print("  02_watermarked.png     → 含水印图")
    print("  03_residual_wm_x20.png → 水印残差（×20）")
    print("  04_attacked.png        → 扰动后图")
    print("  05_residual_noise_x20.png → 扰动残差（×20）")