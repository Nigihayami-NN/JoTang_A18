import os
import numpy as np

# ⚠️ 指向你现在报错的数据集标签路径
LBL_DIR = '/CustomDataSet/labels/train'


def filter_labels():
    files = [f for f in os.listdir(LBL_DIR) if f.endswith('.txt')]
    print(f"🧹 正在对 {len(files)} 个标签文件进行滤毒...")

    removed_boxes = 0
    total_files_fixed = 0

    for filename in files:
        path = os.path.join(LBL_DIR, filename)
        with open(path, 'r') as f:
            lines = f.readlines()

        valid_lines = []
        file_changed = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue

            try:
                # 解析 OBB (cls, x1, y1, x2, y2, x3, y3, x4, y4)
                cls = parts[0]
                coords = np.array([float(x) for x in parts[1:9]]).reshape(4, 2)

                # --- 核心滤毒逻辑 ---
                # 1. 检查坐标是否在合理范围内 (0~1 归一化)
                # 如果你的坐标是 0~640 的像素值，请根据实际情况调整判断
                if np.any(coords < -0.1) or np.any(coords > 1.1):
                    removed_boxes += 1
                    file_changed = True
                    continue

                # 2. 计算鞋带面积 (Shoelace Area)
                area = 0.5 * np.abs(np.dot(coords[:, 0], np.roll(coords[:, 1], 1)) -
                                    np.dot(coords[:, 1], np.roll(coords[:, 0], 1)))

                # 3. 过滤掉极小框 (面积小于 1e-5，相对于 1.0 的归一化面积)
                if area < 0.00001:
                    removed_boxes += 1
                    file_changed = True
                    continue

                # 4. 检查是否包含 NaN 或 Inf
                if not np.isfinite(coords).all():
                    removed_boxes += 1
                    file_changed = True
                    continue

                valid_lines.append(line)

            except:
                removed_boxes += 1
                file_changed = True
                continue

        if file_changed:
            with open(path, 'w') as f:
                f.writelines(valid_lines)
            total_files_fixed += 1

    print(f"\n✨ 滤毒完成！")
    print(f"🗑️ 剔除了 {removed_boxes} 个坏框。")
    print(f"📝 修复了 {total_files_fixed} 个标签文件。")


if __name__ == '__main__':
    filter_labels()