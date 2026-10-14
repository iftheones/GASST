import numpy as np
from utils.dataset import load_mat_hsi

# 分析数据集质量和类别分布
datasets = ['pu', 'ip', 'houston13', 'whuhh']

for dataset_name in datasets:
    print(f"\n=== 分析数据集: {dataset_name} ===")
    
    # 加载数据
    image, gt, labels = load_mat_hsi(dataset_name, "./datasets")
    
    # 基本信息
    print(f"图像形状: {image.shape}")
    print(f"标签形状: {gt.shape}")
    print(f"类别数量: {len(labels)}")
    
    # 检查缺失值
    has_nan = np.isnan(image).any()
    print(f"是否存在缺失值: {has_nan}")
    
    if has_nan:
        nan_count = np.isnan(image).sum()
        total_pixels = image.size
        print(f"缺失值比例: {nan_count / total_pixels:.6f}")
    
    # 类别分布
    print("\n类别分布:")
    class_counts = {}
    for i in range(len(labels)):
        count = np.sum(gt == i)
        class_counts[i] = count
        print(f"  {labels[i]}: {count} 像素")
    
    # 计算类别不平衡情况
    min_class = min(class_counts.values())
    max_class = max(class_counts.values())
    print(f"\n类别不平衡情况:")
    print(f"  最小类别: {min_class} 像素")
    print(f"  最大类别: {max_class} 像素")
    print(f"  不平衡比例: {max_class / min_class:.2f}:1")
    
    # 有效像素比例
    total_pixels = gt.size
    valid_pixels = np.sum(gt >= 0)
    print(f"\n有效像素比例: {valid_pixels / total_pixels:.4f}")
    print(f"无效像素数量: {total_pixels - valid_pixels}")
    
    # 数据范围
    print(f"\n数据范围:")
    print(f"  最小值: {np.min(image):.4f}")
    print(f"  最大值: {np.max(image):.4f}")
    print(f"  平均值: {np.mean(image):.4f}")
    print(f"  标准差: {np.std(image):.4f}")

print("\n=== 数据分析完成 ===")
