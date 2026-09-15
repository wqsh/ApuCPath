import numpy as np
import cv2
from pathlib import Path
from skimage.draw import disk, polygon


def create_simulated_he_image(shape=(256, 256), num_regions=5):
    """生成模拟H&E风格的RGB图像（带有随机纹理和染色效果）"""
    img = np.zeros((*shape, 3), dtype=np.uint8)
    # 基底：粉色/紫色调（模拟H&E背景）
    img[:, :] = [200, 180, 220]  # 浅紫粉色

    # 添加一些随机噪点模拟组织纹理
    noise = np.random.randint(0, 30, (*shape, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)

    # 随机绘制一些椭圆形/多边形区域，模拟不同组织区域
    for _ in range(num_regions):
        center_y = np.random.randint(50, shape[0] - 50)
        center_x = np.random.randint(50, shape[1] - 50)
        radius_y = np.random.randint(20, 60)
        radius_x = np.random.randint(20, 60)
        angle = np.random.randint(0, 360)

        # 创建椭圆掩码
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), angle, 0, 360, 1, -1)

        # 区域内部颜色：偏蓝或偏红（模拟不同类型组织）
        color = [np.random.randint(100, 200),
                 np.random.randint(100, 150),
                 np.random.randint(100, 200)]
        img[mask == 1] = (img[mask == 1] * 0.6 + np.array(color) * 0.4).astype(np.uint8)

        # 添加模拟细胞核的小点（小圆点）
        num_nuclei = np.random.randint(50, 200)
        for _ in range(num_nuclei):
            y = np.random.randint(center_y - radius_y, center_y + radius_y)
            x = np.random.randint(center_x - radius_x, center_x + radius_x)
            if 0 <= y < shape[0] and 0 <= x < shape[1] and mask[y, x]:
                rr, cc = disk((y, x), radius=1, shape=shape)
                img[rr, cc] = [30, 30, 80]  # 深蓝色核

    # 添加一些模糊和轻微亮度变化
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def create_simulated_mask(shape=(256, 256), num_classes=3):
    """生成与模拟图像对应的语义分割掩码（0背景，1上皮，2间质）"""
    mask = np.zeros(shape, dtype=np.uint8)

    # 随机生成一些多边形区域作为上皮组织（类1）
    num_epi = np.random.randint(2, 4)
    for _ in range(num_epi):
        vertices = []
        center = (np.random.randint(50, shape[0] - 50), np.random.randint(50, shape[1] - 50))
        radius = np.random.randint(30, 70)
        # 生成不规则多边形
        for angle in np.linspace(0, 2 * np.pi, np.random.randint(5, 8)):
            r = radius + np.random.randint(-10, 10)
            x = int(center[1] + r * np.cos(angle))
            y = int(center[0] + r * np.sin(angle))
            vertices.append([x, y])
        rr, cc = polygon([p[1] for p in vertices], [p[0] for p in vertices], shape=shape)
        mask[rr, cc] = 1

    # 随机生成一些区域作为间质组织（类2），避免与上皮重叠
    num_str = np.random.randint(2, 4)
    for _ in range(num_str):
        vertices = []
        center = (np.random.randint(50, shape[0] - 50), np.random.randint(50, shape[1] - 50))
        radius = np.random.randint(30, 70)
        for angle in np.linspace(0, 2 * np.pi, np.random.randint(5, 8)):
            r = radius + np.random.randint(-10, 10)
            x = int(center[1] + r * np.cos(angle))
            y = int(center[0] + r * np.sin(angle))
            vertices.append([x, y])
        rr, cc = polygon([p[1] for p in vertices], [p[0] for p in vertices], shape=shape)
        # 仅当该位置不是上皮时才设为间质
        mask[(rr, cc)] = np.where(mask[rr, cc] == 0, 2, mask[rr, cc])

    # 可选：添加一些小的背景空洞
    holes = np.random.randint(0, 10, shape) < 0.01
    mask[holes] = 0

    return mask


def generate_dataset(output_dir="sample_data", num_samples=3, image_size=(256, 256)):
    """生成完整的模拟数据集（图像+掩码）"""
    output_dir = Path(output_dir)
    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    for idx in range(num_samples):
        # 生成图像和掩码
        img = create_simulated_he_image(shape=image_size)
        mask = create_simulated_mask(shape=image_size, num_classes=3)

        # 保存到 train 和 val 各复制一份（简化，实际应分开）
        for split in ["train", "val"]:
            img_path = output_dir / split / "images" / f"sample_{idx:03d}.png"
            mask_path = output_dir / split / "masks" / f"sample_{idx:03d}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_path), mask)

    print(f"成功生成 {num_samples} 组示例数据，保存在 {output_dir}")
    print("目录结构：")
    print("  sample_data/train/images/*.png")
    print("  sample_data/train/masks/*.png")
    print("  sample_data/val/images/*.png")
    print("  sample_data/val/masks/*.png")
    print("掩码像素值含义：0=背景, 1=上皮组织, 2=间质组织")


if __name__ == "__main__":
    generate_dataset()