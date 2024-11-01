#填充纯黑图片至36
import os
from PIL import Image

# 指定包含小文件夹的目录
folder_path = r"duocengMUBA_rem_256_312/duocengMUBA_rem"

# 遍历目录下的每个小文件夹
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
# 判断是否是文件夹
    if os.path.isdir(subfolder_path):
# 列出文件夹中的图像文件
        image_files = [f for f in os.listdir(subfolder_path) if f.endswith(".png") or f.endswith(".jpg")]
# 计算图像数量
    num_images = len(image_files)
# 如果图像数量小于36，进行填充
    if num_images < 36:
        num_to_fill = 36 - num_images
        for i in range(num_to_fill):
        # 创建一张256x312的黑色图像
            black_image = Image.new('L', (256, 312), color=0)
# 保存黑色图像到小文件夹中
            black_image.save(os.path.join(subfolder_path, f"black_image_{i}.png"))
print(subfolder_path,"已完成")

print("填充完成。")#这里图片尺寸你对一下