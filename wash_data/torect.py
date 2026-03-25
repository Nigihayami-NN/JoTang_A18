import os
import cv2
import numpy as np
import glob

# ⚠️ 根据你之前的物理切边逻辑，现在的图应该是 640x512
IMG_W = 640.0
IMG_H = 512.0

LBL_DIR = 'D:/Python/fordata/CustomDataSet/labels/train'


def order_points_clockwise(pts):
    """将 4 个点按顺时针排序，强制左上角为 P1"""
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    pts_sorted = pts[sorted_idx]

    sums = pts_sorted[:, 0] + pts_sorted[:, 1]
    top_left_idx = np.argmin(sums)
    pts_final = np.roll(pts_sorted, shift=-top_left_idx, axis=0)
    return pts_final


def clean_obb_labels_v2(label_dir):
    txt_files = glob.glob(os.path.join(label_dir, '*.txt'))
    print(f"🚀 开始在物理像素空间 ({IMG_W}x{IMG_H}) 下重塑矩形...")

    fixed_count = 0
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue

            cls_id = parts[0]
            # 1. 读取归一化坐标
            coords_norm = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)

            # 2. 【核心修复】：还原到真实的物理像素坐标！！！
            coords_pixel = coords_norm.copy()
            coords_pixel[:, 0] *= IMG_W
            coords_pixel[:, 1] *= IMG_H

            # 3. 在物理空间计算绝对完美的 90度 最小外接矩形
            rect = cv2.minAreaRect(coords_pixel)
            box_pixel = cv2.boxPoints(rect)

            # 4. 顺时针排序
            box_ordered_pixel = order_points_clockwise(box_pixel)

            # 5. 重新归一化回 0~1 之间
            box_norm = box_ordered_pixel.copy()
            box_norm[:, 0] /= IMG_W
            box_norm[:, 1] /= IMG_H

            # 6. 安全截断
            box_clamped = np.clip(box_norm, 0.0, 1.0)

            new_coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in box_clamped])
            new_lines.append(f"{cls_id} {new_coords_str}\n")

        with open(txt_file, 'w') as f:
            f.writelines(new_lines)

        fixed_count += 1

    print(f"✅ V2 清洗完毕！成功处理了 {fixed_count} 个文件。框现在是绝对的长方形了！")


if __name__ == '__main__':
    clean_obb_labels_v2(LBL_DIR)

    # 记得把验证集也洗了
    VAL_DIR = 'D:/Python/fordata/CustomDataSet/labels/val'  # 确保这个路径正确
    if os.path.exists(VAL_DIR):
        clean_obb_labels_v2(VAL_DIR)