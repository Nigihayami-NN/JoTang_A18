import cv2
import numpy as np
import os
from pathlib import Path

def verify_alignment_by_features(rgb_path, ir_path):
    """通过特征点匹配验证对齐程度"""
    rgb = cv2.imread(rgb_path)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    
    # 转换RGB为灰度
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    
    # 调整尺寸一致
    ir_resized = cv2.resize(ir, (rgb_gray.shape[1], rgb_gray.shape[0]))
    
    # 使用ORB特征检测
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(rgb_gray, None)
    kp2, des2 = orb.detectAndCompute(ir_resized, None)
    
    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # 计算匹配点的偏移
    if len(matches) > 0:
        offsets_x = []
        offsets_y = []
        
        for match in matches[:50]:  # 检查前50个匹配点
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            offsets_x.append(abs(pt1[0] - pt2[0]))
            offsets_y.append(abs(pt1[1] - pt2[1]))
        
        avg_offset_x = np.mean(offsets_x)
        avg_offset_y = np.mean(offsets_y)
        
        print(f"平均X轴偏移: {avg_offset_x:.2f} 像素")
        print(f"平均Y轴偏移: {avg_offset_y:.2f} 像素")
        print(f"匹配点对数量: {len(matches)}")
        
        # 判断是否对齐（阈值可调整）
        if avg_offset_x < 5 and avg_offset_y < 5:
            print("✓ 图像对齐良好")
            return True
        else:
            print("✗ 图像存在偏移，需要校正")
            return False
    
    return False



def align_with_feature_matching(rgb_path, ir_path, output_dir='align', file_index=1):
    os.makedirs(output_dir, exist_ok=True)
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    # 将 RGB 转为灰度用于特征匹配
    rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # 统一尺寸
    h, w = rgb_img.shape[:2]
    ir_img = cv2.resize(ir_img, (w, h), interpolation=cv2.INTER_LINEAR)

    # ORB 特征匹配
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(rgb_gray, None)
    kp2, des2 = orb.detectAndCompute(ir_img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    align_success=False
    if len(matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is not None:
            ir_aligned = cv2.warpPerspective(ir_img, M, (w, h))
            align_success = True
        else:
            print("计算单应性矩阵失败，使用简单缩放对齐")
            ir_aligned = ir_img
    else:
        print("特征点不足，使用简单缩放对齐")
        ir_aligned = ir_img

    # 将 RGB 图像 (3通道) 与对齐后的特征图/IR图 (1通道) 拼接为4通道矩阵
    if len(ir_aligned.shape) == 2:
        ir_aligned_expanded = np.expand_dims(ir_aligned, axis=-1)
    else:
        ir_aligned_expanded = ir_aligned

    merged_4_channel = np.concatenate((rgb_img, ir_aligned_expanded), axis=-1)

    # 1. 保存拼接后的 4 通道矩阵（保存为 .npy 格式）
    aligned_npy_path = os.path.join(output_dir, f"{file_index}.npy")
    np.save(aligned_npy_path, merged_4_channel)
    
    print(f"配准合并完成，4通道矩阵已保存至：{aligned_npy_path}")
    
    return align_success




def process_single_folder(folder_dir):
    """遍历单个文件夹并验证图片对齐（假定按名称排序后，相邻的两张图片分别为RGB和TR/IR，即两两成对）"""
    folder_path = Path(folder_dir)
    
    # 获取文件夹中所有的图片文件，并按名称排序保证顺序
    valid_extensions = {'.jpg', '.png', '.jpeg', '.bmp'}
    all_files = sorted([f for f in folder_path.glob('*') if f.is_file() and f.suffix.lower() in valid_extensions])
    
    if len(all_files) % 2 != 0:
        print(f"警告：文件夹中的图片数量 ({len(all_files)}) 为奇数，最后一张图片将无法配对！")
        
    success_count = 0
    fail_count = 0
    
    # 每次步进为2，取出相邻的两张图片（假定前者为RGB，后者为TR/IR，如果顺序相反不影响对比结果）
    for i in range(0, len(all_files) - 1, 2):
        file_index = (i // 2) + 1  # 顺序编号，从1开始
        rgb_path = all_files[i]
        ir_path = all_files[i+1]
        
        print(f"\n--- 正在配对处理: {rgb_path.name} & {ir_path.name} ---")
        try:
            # 调用新修改的配准函数，保存到默认的 align 文件夹，并传入编号
            is_aligned = align_with_feature_matching(str(rgb_path), str(ir_path), output_dir='align', file_index=file_index)
            if is_aligned:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"处理文件对 {rgb_path.name} & {ir_path.name} 时出错: {e}")
            
    print(f"\n=== 处理完成 ===")
    print(f"处理总计: {success_count + fail_count} 对图片")
    #print(f"对齐良好: {success_count} 对")
    #print(f"存在偏移: {fail_count} 对")

if __name__ == "__main__":
    # 请在此处修改为您实际混合存放图片的文件夹路径
    mixed_image_folder = "D:/APP/Vehicules1024"
    
    if os.path.exists(mixed_image_folder):
        process_single_folder(mixed_image_folder)
    else:
        print(f"找不到文件夹: {mixed_image_folder}，请填入实际的文件夹路径再运行脚本")