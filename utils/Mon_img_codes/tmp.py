import os
import random
import shutil

# 源目录和目标目录
src_dir = "/media/4TB/GJ/TS/splitedData/Mon_data/train/jf"
dst_dir = "/media/4TB/GJ/TS/splitedData/Mon_data/test/jf"

os.makedirs(dst_dir, exist_ok=True)

# 获取所有 png 文件
png_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".png")]

# 随机选择25个（如果不足25个，则取全部）
sample_files = random.sample(png_files, min(40, len(png_files)))

# 移动文件
for f in sample_files:
    shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

print(f"已移动 {len(sample_files)} 个文件到 {dst_dir}")
