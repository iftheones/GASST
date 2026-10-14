import numpy as np
from scipy import io
import os

# 简化版数据分析，不依赖 PyTorch

def load_mat_hsi_simple(dataset_name, dataset_dir):
    """ 简化版数据加载，不包含 PyTorch 相关代码 """
    available_sets = [
        'pu', 'sa', 'ip', 'whulk', 'whuhc', 'whuhh', 
        'houston13', 'houston18', 'QUH_TDW', 'QUH_QY', 'QUH_PA'
    ]
    assert dataset_name in available_sets, f"dataset should be one of {available_sets}"

    image = None
    gt = None
    labels = None

    # 简化的数据集加载逻辑
    if dataset_name == 'pu':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU.mat"))['paviaU']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU_gt.mat"))['paviaU_gt']
        labels = [
            "Asphalt", "Meadows", "Gravel", "Trees", 
            "Painted metal sheets", "Bare Soil", "Bitumen", 
            "Self-Blocking Bricks", "Shadows"
        ]
    elif dataset_name == 'ip':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Indian_pines_corrected.mat"))['indian_pines_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Indian_pines_gt.mat"))['indian_pines_gt']
        labels = [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", 
            "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", 
            "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", 
            "Soybean-clean", "Wheat", "Woods", 
            "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
        ]
    elif dataset_name == 'houston13':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "GRSS2013.mat"))['GRSS2013']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "GRSS2013_gt.mat"))['GRSS2013_gt']
        labels = [
            "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", 
            "Soil", "Water", "Residential", "Commercial", 
            "Road", "Highway", "Railway", "Parking Lot 1", 
            "Parking Lot 2", "Tennis Court", "Running Track"
        ]
    elif dataset_name == 'whuhh':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HongHu.mat"))['WHU_Hi_HongHu']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HongHu_gt.mat"))['WHU_Hi_HongHu_gt']
        labels = [
            'Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 
            'Rape', 'Chinese cabbage', 'Pakchoi', 'Cabbage', 'Tuber mustard', 
            'Brassica parachinensis', 'Brassica chinensis', 'Small Brassica chinensis', 
            'Lactuca sativa', 'Celtuce', 'Film covered lettuce', 'Romaine lettuce', 
            'Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree'
        ]
    else:
        print(f"暂不支持数据集: {dataset_name}")
        return None, None, None

    # 基本预处理
    image = np.asarray(image, dtype=np.float32)
    gt = gt.astype('int') - 1  # 从 0 开始索引

    return image, gt, labels

# 分析数据集
datasets = ['pu', 'ip', 'houston13', 'whuhh']

for dataset_name in datasets:
    print(f"\n=== 分析数据集: {dataset_name} ===")
    
    # 加载数据
    image, gt, labels = load_mat_hsi_simple(dataset_name, "./datasets")
    
    if image is None:
        continue
    
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
