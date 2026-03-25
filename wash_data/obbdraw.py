import cv2
import os
import numpy as np

# === 1. 路径配置 (根据你的 CustomDataSet 结构) ===
dataset_root = 'CustomDataSet'
img_path = os.path.join(dataset_root, 'images/train/00123_co.jpg')
label_path = os.path.join(dataset_root, 'labels/train/00123_co.txt')
save_path = os.path.join(dataset_root, 'check_obb_00123.jpg')


# === 2. 执行检查 ===
def check_specific_obb():
    # 检查文件
    if not os.path.exists(img_path):
        print(f"错误: 找不到图片 {img_path}")
        return
    if not os.path.exists(label_path):
        print(f"错误: 找不到标签 {label_path}")
        return

    # 读取图片
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"正在处理: {img_path} | 分辨率: {w}x{h}")

    # 读取 OBB 标签
    with open(label_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue  # OBB 至少需要 class + 8个坐标点

            # 解析格式: cls x1 y1 x2 y2 x3 y3 x4 y4 (假设已归一化)
            cls = parts[0]
            coords = list(map(float, parts[1:]))

            # 还原坐标并转换为像素点整型
            pts = []
            for i in range(0, 8, 2):
                px = int(coords[i] * w)
                py = int(coords[i + 1] * h)
                pts.append([px, py])

            # 转换为 OpenCV 要求的格式 (numpy array)
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # 画旋转框 (多边形)
            # True 表示封闭，绿色 (0, 255, 0)，粗细 2
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # 标上类别 (在第一个点 x1, y1 处标文字)
            cv2.putText(img, f"cls:{cls}", (pts[0][0][0], pts[0][0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 保存结果
    cv2.imwrite(save_path, img)
    print(f"成功！OBB 检查结果已保存至: {save_path}")


# 执行
check_specific_obb()