import torch.optim as optim


def load_scheduler(model_name, model, epochs=500):
    optimizer, scheduler = None, None
    
    if model_name == 'cnn2d':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1)

    elif model_name == 'GASST' or model_name == 'CDSFormer':
        # 改进的优化器配置，适用于Transformer模型
        optimizer = optim.AdamW(model.parameters(), 
                              lr=0.0005,  # 适当提高初始学习率
                              weight_decay=0.0001,  # 增加权重衰减，减少过拟合
                              betas=(0.9, 0.999),
                              eps=1e-08)
        
        # 使用余弦退火学习率调度器，更适合Transformer模型
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,  # 最大迭代次数
            eta_min=1e-6,  # 最小学习率
            last_epoch=-1
        )
        
        # 或者使用带 warmup 的学习率调度器
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer, 
        #     max_lr=0.001, 
        #     epochs=epochs, 
        #     steps_per_epoch=100  # 需要根据实际batch数量调整
        # )

    return optimizer, scheduler


