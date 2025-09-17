
import cv2
import numpy as np
import os

# 输入输出路径
input_folder = r'C:\Users\Administrator\Desktop\111'
output_folder = r'C:\Users\Administrator\Desktop\111\Bi'
os.makedirs(output_folder, exist_ok=True)

# 红色阈值范围（BGR）
lower_red = np.array([0, 0, 150])
upper_red = np.array([80, 80, 255])

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)
        if img is None:
            continue

        # 提取红色区域掩码
        red_mask = cv2.inRange(img, lower_red, upper_red)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化一个黑色背景图（与原图同尺寸）
        result_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 提取每个红框区域并放入 result_mask 中
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            result_mask[y:y+h, x:x+w] = 255  # 将红框区域设为白

        # 原图转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 只保留红框内的像素内容，其他设为 0（背景）
        roi_pixels = cv2.bitwise_and(gray, gray, mask=result_mask)

        # 对保留区域进行 Otsu 二值化
        _, binary = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 保存结果
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_binary.png")
        cv2.imwrite(output_path, binary)

print("全部图像处理完成 ✅")
