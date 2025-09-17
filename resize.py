from PIL import Image
import os

# 设置输入输出文件夹路径
input_folder = r'C:\Users\Administrator\Desktop\dataset5jpg'
output_folder = os.path.join(input_folder, 'resized')
os.makedirs(output_folder, exist_ok=True)

# 目标尺寸
target_size = (640, 512)  # (width, height)

# 遍历图像文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        img_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with Image.open(img_path) as img:
            resized_img = img.resize(target_size, Image.ANTIALIAS)
            resized_img.save(output_path)

print("所有图像已调整为 512x640 大小并保存到:", output_folder)
