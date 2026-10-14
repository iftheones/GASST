import torch
from models.get_model import get_model

# 测试GASST模型的初始化和前向传播

def verify_model():
    print("开始验证GASST模型实现...")
    
    # 模型参数
    model_name = 'GASST'
    dataset_name = 'pu'  # 以Pavia University数据集为例
    kernel_size = 3
    ps = 2  # patch size
    k = 0.5  # Token Selective Attention的k参数
    group_num = 4
    emb_dim = 64
    
    # 初始化模型
    print("正在初始化GASST模型...")
    try:
        model = get_model(model_name, dataset_name, kernel_size, ps, k, group_num, emb_dim)
        print("模型初始化成功！")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return False
    
    # 测试前向传播
    print("测试模型前向传播...")
    try:
        # 创建一个模拟的高光谱图像输入 (batch_size, 1, channels, height, width)
        # 与训练输入一致：PCA 后为 30 个通道
        input_tensor = torch.randn(2, 1, 30, 10, 10)  # 2个样本，103个通道，10x10的空间维度
        output = model(input_tensor)
        print(f"前向传播成功！输出形状: {output.shape}")
        print("GASST模型验证通过！")
        return True
    except Exception as e:
        print(f"前向传播失败: {e}")
        return False

if __name__ == "__main__":
    verify_model()
