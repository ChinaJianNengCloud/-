import os
import shutil

# 源文件夹路径
source_dir = '源文件夹路径'
# 目标文件夹路径
target_dir = '目标文件夹路径'

# 创建目标文件夹
os.makedirs(target_dir, exist_ok=True)

# 遍历源文件夹中的所有子文件夹
for foldername in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, foldername)
    
    # 检查是否为文件夹
    if os.path.isdir(folder_path):
        # 获取所有 PNG 文件
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
        # 如果 PNG 文件数量大于或等于 36，则复制前 36 张图片
        if len(png_files) >= 36:
            # 创建新文件夹
            new_folder_path = os.path.join(target_dir, foldername)
            os.makedirs(new_folder_path, exist_ok=True)
            
            # 复制前 36 张 PNG 图片
            for png_file in png_files[:36]:
                src_file = os.path.join(folder_path, png_file)
                dst_file = os.path.join(new_folder_path, png_file)
                shutil.copy(src_file, dst_file)

print("图片复制完成！")
