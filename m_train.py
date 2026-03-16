import torch
import os
from ultralytics import YOLO


def transplant_and_train():
    import datetime
    print(f"[{datetime.datetime.now()}] 开始执行训练脚本")
    
    # ==========================================
    # 1. 核心路径配置
    # ==========================================
    YAML_PATH = "m-test.yaml"  # 你另存为的 M 版配置
    # DATA_YAML = "test_bus.yaml"
    DATA_YAML = "CustomDataSet.yaml"

    # 【关键修改】：换成 M 级官方权重
    OFFICIAL_PT = "yolo26m.pt"
    INIT_PT = "weights/rgbt_v26M_init.pt"
    
    print(f"[{datetime.datetime.now()}] 配置信息：")
    print(f"  - 配置文件: {YAML_PATH}")
    print(f"  - 数据集: {DATA_YAML}")
    print(f"  - 官方权重: {OFFICIAL_PT}")
    print(f"  - 初始化权重: {INIT_PT}")

    # ==========================================
    # 2. 权重移植逻辑 (一键无缝转换)
    # ==========================================
    if not os.path.exists(INIT_PT):
        print("正在构建 11 类非对称双流预训练权重...")
        # 初始化空白的定制模型
        custom_model = YOLO(YAML_PATH)
        # 加载官方模型
        official_model = YOLO(OFFICIAL_PT)

        c_state = custom_model.model.state_dict()
        o_state = official_model.model.state_dict()

        # 映射表: RGB(1-10)->官方(0-9); Thermal(11-20)->官方(0-9)
        layer_map = {str(i): str(i - 1) for i in range(1, 11)}
        layer_map.update({str(i): str(i - 11) for i in range(11, 21)})

        count = 0
        for k, v in c_state.items():
            p = k.split('.')
            if len(p) > 1 and p[1] in layer_map:
                off_k = k.replace(f"model.{p[1]}.", f"model.{layer_map[p[1]]}.")
                if off_k in o_state:
                    # 针对红外第一层的 1 通道特殊处理 (3通道求均值)
                    if p[1] == '11' and 'conv.weight' in k:
                        c_state[k] = o_state[off_k].mean(dim=1, keepdim=True)
                        count += 1
                    # 常规拷贝
                    elif c_state[k].shape == o_state[off_k].shape:
                        c_state[k] = o_state[off_k]
                        count += 1

        # 保存带血包的新权重
        custom_model.model.load_state_dict(c_state)
        custom_model.save(INIT_PT)
        print(f"权重移植成功！共映射了 {count} 个张量，保存至 {INIT_PT}")
    else:
        print(f"发现已存在的移植权重 {INIT_PT}，直接加载。")

    # ==========================================
    # 3. 启动 RTX 4090 正式训练
    # ==========================================
    print("启动训练")

    # 重新加载带有预训练权重的模型
    model = YOLO(INIT_PT)

    # 核心训练参数 (专门针对无人机 RGBT 和大算力优化)
    results = model.train(
        data=DATA_YAML,
        imgsz=1024,  # 训练分辨率 (4090 的黄金尺寸)
        epochs=150,  # 国家级比赛必须跑到 300 轮
        batch=16,  # 4090 可以尝试 16，如果 OOM 则改为 8 且设置 accumulate=2
        device='cpu',  # 指定CPU，因为没有可用的CUDA设备
        amp=True,  # 混合精度 (加速且省显存)

        # --- 针对 4 通道的必须设置 ---
        hsv_h=0.0,  # 绝对禁用色调增强 (防止 cv2 崩溃)
        hsv_s=0.0,  # 绝对禁用饱和度增强
        hsv_v=0.0,  # 绝对禁用亮度增强

        # --- 针对小目标的高级增强 ---
        mosaic=1.0,  # 100% 开启马赛克增强
        mixup=0.15,  # 引入少量 Mixup 增加背景鲁棒性
        multi_scale=True,  # 开启多尺度训练 (极其重要！让模型适应各种高度的车辆)

        # mosaic=0.0,  # 关闭马赛克
        # mixup=0.0,  # 关闭 Mixup
        # copy_paste=0.0,  # 关闭复制粘贴

        # --- 优化器与日志 ---
        optimizer='AdamW',  # Transformer 架构的首选优化器
        lr0=0.001,  # AdamW 初始学习率建议比 SGD 小一点
        warmup_epochs=5,  # 给 Swin 层 5 个 Epoch 慢慢预热，防止初期梯度爆炸

        project="Drone_RGBT_Runs",
        name="YOLO26m-test",
        save_period=10  # 每 10 轮保存一次权重，防止服务器意外断电
    )
    print("训练结束！")


if __name__ == "__main__":
    transplant_and_train()