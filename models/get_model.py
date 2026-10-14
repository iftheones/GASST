from .GASST import *
from .cnn2d import *

def get_model(model_name, dataset_name, kernel_size, ps, k, group_num, emb_dim, depth=6, drop_rate=0.1, num_heads=8, use_pca=True, pc=30):
    if model_name == 'GASST':
        model = GASST(dataset_name, kernel_size, ps, k, group_num, emb_dim, depth=depth, drop_rate=drop_rate, num_heads=num_heads)
    
    elif model_name == 'CDSFormer':  # 新增CDSFormer模型选项
        # 由于我们已经在DSFormer类中集成了STEB模块，这里可以直接使用DSFormer类
        # 这样在命令行中可以通过 --model CDSFormer 来调用改进后的模型
        model = GASST(dataset_name, kernel_size, ps, k, group_num, emb_dim, depth=depth, drop_rate=drop_rate, num_heads=num_heads)

    elif model_name == 'cnn2d':
        model = cnn2d(dataset_name)
    
    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return model

