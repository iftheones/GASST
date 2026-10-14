import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import grouper, sliding_window, count_sliding_window
import matplotlib.pyplot as plt

def train(network, optimizer, criterion, train_loader, val_loader, epoch, saving_path, device, scheduler=None):

    best_acc = -0.1
    losses = []
    patience = 30  # 早停耐心值
    patience_counter = 0
    
    for e in tqdm(range(1, epoch+1), desc=""):
        network.train()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = network(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        
        # 每10个epoch或第一个epoch验证一次
        if e % 10 == 0 or e == 1:
            mean_losses = np.mean(losses)
            
            # 验证模型
            val_acc = validation(network, val_loader, device)
            
            train_info = "train at epoch {}/{}, loss={:.6f}, val_acc={:.6f}"
            train_info = train_info.format(e, epoch, mean_losses, val_acc)
            tqdm.write(train_info)
            
            # 早停策略
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                save_checkpoint(network, True, saving_path, epoch=e, acc=best_acc)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"Early stopping at epoch {e}, best_val_acc={best_acc:.6f}")
                    break
            
            losses = []
        else:
            losses = []

def train_1(network, optimizer, criterion, train_loader, val_loader, epoch, saving_path, device, scheduler=None):

    best_acc = -0.1
    losses = []
    # X = X.to(device)
    for e in tqdm(range(1, epoch + 1), desc=""):
        network.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.squeeze(1)
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            # outputs = network(images,X)
            outputs = network(images)
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
        if e % 10 == 0 or e == 1:
            mean_losses = np.mean(losses)
            train_info = "train at epoch {}/{}, loss={:.6f}"
            train_info = train_info.format(e, epoch, mean_losses)
            tqdm.write(train_info)
            losses = []
        else:
            losses = []
        # val_acc = validation(network, val_loader, device)
        #
        # if scheduler is not None:
        #     scheduler.step()
        #
        # is_best = val_acc >= best_acc
        # best_acc = max(val_acc, best_acc)
        # save_checkpoint(network, is_best, saving_path, epoch=e, acc=best_acc)

def validation(network, val_loader, device):
    num_correct = 0.
    total_num = 0.
    network.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)

            outputs = network(images)
            _, predicted = torch.max(outputs, dim=1)
            for pred, target in zip(predicted, targets):
                num_correct = num_correct + (pred.item() == target.item())
                total_num = total_num + 1
    overall_acc = num_correct / total_num
    return overall_acc


def test(network, model_dir, image, patch_size, n_classes, device):
    # network.load_state_dict(torch.load(model_dir + "/model_best.pth"))
    network.eval()

    patch_size = patch_size
    batch_size = 1024
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2

    # pad the image
    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    probs = np.zeros(image.shape[:2] + (n_classes, ))

    iterations = count_sliding_window(image, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                      total=iterations,
                      desc="inference on the HSI"):
        with torch.no_grad():
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose((0, 3, 1, 2))
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)

            output = network(data)
            output = output.to('cpu').numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out
    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]


def test_ensemble(networks, model_dirs, image, patch_size, n_classes, device):
    """
    集成测试函数，融合多个模型的预测结果
    :param networks: 模型列表
    :param model_dirs: 模型目录列表
    :param image: 输入图像
    :param patch_size: 补丁大小
    :param n_classes: 类别数量
    :param device: 设备
    :return: 集成预测结果
    """
    patch_size = patch_size
    batch_size = 1024
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2

    # pad the image
    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    # 初始化概率数组，用于存储所有模型的平均概率
    ensemble_probs = np.zeros(image.shape[:2] + (n_classes, ))

    # 对每个模型进行预测
    for i, network in enumerate(networks):
        print(f"\n使用模型 {i+1}/{len(networks)} 进行预测...")
        # network.load_state_dict(torch.load(model_dirs[i] + "/model_best.pth"))
        network.eval()
        
        # 每个模型的概率预测
        model_probs = np.zeros(image.shape[:2] + (n_classes, ))
        
        iterations = count_sliding_window(image, window_size=window_size) // batch_size
        for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                          total=iterations,
                          desc=f"Model {i+1} inference"):
            with torch.no_grad():
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose((0, 3, 1, 2))
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

                indices = [b[1:] for b in batch]
                data = data.to(device)

                output = network(data)
                output = output.to('cpu').numpy()

                for (x, y, w, h), out in zip(indices, output):
                    model_probs[x + w // 2, y + h // 2] += out
        
        # 将当前模型的预测结果添加到集成结果中
        ensemble_probs += model_probs
    
    # 计算平均概率
    ensemble_probs /= len(networks)
    
    return ensemble_probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]

def test_1(network, model_dir, image, patch_size, n_classes, device):
    # network.load_state_dict(torch.load(model_dir + "/model_best.pth"))
    network.eval()

    patch_size = patch_size
    batch_size = 1024
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2
    # X = X.to(device)

    # pad the image
    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    probs = np.zeros(image.shape[:2] + (n_classes, ))

    iterations = count_sliding_window(image, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                      total=iterations,
                      desc="inference on the HSI"):
        with torch.no_grad():
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose((0, 3, 1, 2))
            data = torch.from_numpy(data)
            # data = data.unsqueeze(1)   # ss_mamba b,c,h,w

            indices = [b[1:] for b in batch]
            data = data.to(device)
            # output = network(data,X)

            output = network(data)
            output = output.to('cpu').numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out
    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]

def save_checkpoint(network, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write("epoch = {epoch}: best OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best.pth'))
    else:  # save the ckpt for each 10 epoch
        if kwargs['epoch'] % 10 == 0:
            torch.save(network.state_dict(), os.path.join(saving_path, 'model.pth'))



