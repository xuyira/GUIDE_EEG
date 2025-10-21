import os
import numpy as np
import scipy.io
import random
import torch
import torch as th  

class DelayEmbedder:
    """Delay embedding transformation"""
    
    def __init__(self, device, seq_len, delay, embedding):
        self.device = device
        self.seq_len = seq_len
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None
    
    def pad_to_square(self, x, mask=0):
        """Pads the input tensor x to make it square along the last two dimensions."""
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (0, max_side - rows, 0, max_side - cols)
        x_padded = th.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded
    
    def ts_to_img(self, signal, pad=True, mask=0):
        """
        将时间序列转换为图像
        Args:
            signal: (batch, length, features) - EEG数据
            pad: 是否填充到正方形
        Returns:
            x_image: (batch, features, H, W)
        """
        batch, length, features = signal.shape
        if self.seq_len != length:
            self.seq_len = length
        
        x_image = th.zeros((batch, features, self.embedding, self.embedding))
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1
        
        # 处理剩余部分
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1
        
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]
        
        if pad:
            x_image = self.pad_to_square(x_image, mask)
        
        return x_image


def img_to_ts(image_data):
    """
    将图像数据转换为时间序列格式（使用时间延迟嵌入的逆过程）
    
    参数:
        image_data: torch tensor, shape=(batch, 22, 64, 64)
    
    返回:
        time_series: torch tensor, shape=(batch, 1000, 22)
    """
    batch, channels, rows, cols = image_data.shape
    
    # 参数设置
    seq_len = 1000
    delay = 15
    embedding = 64
    
    # 初始化重建的时间序列
    reconstructed_x_time_series = th.zeros((batch, channels, seq_len))
    
    # 重建时间序列（时间延迟嵌入的逆过程）
    for i in range(cols - 1):
        start = i * delay
        end = start + embedding
        reconstructed_x_time_series[:, :, start:end] = image_data[:, :, :, i]
    
    # 处理最后一列（特殊情况）
    start = (cols - 1) * delay
    end = reconstructed_x_time_series[:, :, start:].shape[-1]
    reconstructed_x_time_series[:, :, start:] = image_data[:, :, :end, cols - 1]
    
    # 转换维度: (batch, channels, seq_len) -> (batch, seq_len, channels)
    reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)
    
    return reconstructed_x_time_series


def eegdata_preprocess(testsubj, dataset='bci2a', delay=15, embedding=64, device='cuda'):
    """
    EEG 数据预处理函数
    
    参数:
        testsubj: int, 测试受试者编号 (1-9)
        dataset: str, 数据集名称 ('bci2a' 或 'bci2b')
        delay: int, 时间延迟参数
        embedding: int, 嵌入维度
        device: str, 设备 ('cuda' 或 'cpu')
    
    返回:
        train_data: torch.Tensor, 训练数据图像 (N_train, channels, H, W)
        train_label: torch.Tensor, 训练标签 (N_train,)
        test_data: torch.Tensor, 测试数据图像 (N_test, channels, H, W)
        test_label: torch.Tensor, 测试标签 (N_test,)
    """
    
    print('=' * 80)
    print(f'EEG 数据预处理')
    print(f'数据集: {dataset}')
    print(f'测试受试者: Subject {testsubj}')
    print('=' * 80)
    
    # ========== Step 1: 根据数据集选择数据路径 ==========
    if dataset == 'bci2a':
        data_root = './data/standard_2a_data/'
        num_subjects = 9
        num_classes = 4
    elif dataset == 'bci2b':
        # TODO: 实现 bci2b 的处理逻辑
        raise NotImplementedError("bci2b 数据集处理逻辑尚未实现")
    else:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    # ========== Step 2: 根据 testsubj 划分训练集和测试集 ==========
    
    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []
    
    for subj in range(1, num_subjects + 1):
        # 加载原始训练数据 (T - Training)
        train_file = data_root + f'A0{subj}T.mat'
        train_mat = scipy.io.loadmat(train_file)
        data_T = train_mat['data']  # (time_points, channels, trials)
        label_T = train_mat['label']  # (trials, 1)
        
        # 加载原始测试数据 (E - Evaluation)
        test_file = data_root + f'A0{subj}E.mat'
        test_mat = scipy.io.loadmat(test_file)
        data_E = test_mat['data']  #  (time_points, channels, trials)
        label_E = test_mat['label']  # (trials, 1)
        
        # 转换维度: (time_points, channels, trials) -> (trials, time_points, channels)
        data_T = np.transpose(data_T, (2, 0, 1))  # (trials, time_points, channels)
        label_T = label_T.flatten()               # (trials,) - 改用 flatten 确保是 1D
        
        data_E = np.transpose(data_E, (2, 0, 1))  # (trials, time_points, channels)
        label_E = label_E.flatten()               # (trials,) - 改用 flatten 确保是 1D
        
        # 根据 testsubj 划分
        if subj == testsubj:
            # 当前受试者作为测试集
            test_data_list.append(data_T)
            test_data_list.append(data_E)
            test_label_list.append(label_T)
            test_label_list.append(label_E)
        else:
            # 其他受试者作为训练集
            train_data_list.append(data_T)
            train_data_list.append(data_E)
            train_label_list.append(label_T)
            train_label_list.append(label_E)
    
    # 合并数据
    train_data = np.concatenate(train_data_list, axis=0)  # (N_train, time_points, channels)
    train_label = np.concatenate(train_label_list, axis=0)  # (N_train,)
    test_data = np.concatenate(test_data_list, axis=0)     # (N_test, time_points, channels)
    test_label = np.concatenate(test_label_list, axis=0)   # (N_test,)
    
    print(f'训练集: {train_data.shape}, 标签: {train_label.shape}')
    print(f'测试集: {test_data.shape}, 标签: {test_label.shape}')
    
    # 标签从 1-4 转换为 0-3
    train_label = train_label - 1
    test_label = test_label - 1
    
    # ========== Step 3: 归一化[-1,1] ==========
    train_min = np.min(train_data)
    train_max = np.max(train_data)

    train_data = 2 * (train_data - train_min) / (train_max - train_min) - 1
    test_data = np.clip(2 * (test_data - train_min) / (train_max - train_min) - 1, -1, 1)

    print(f"训练集原始范围: [{train_min:.4f}, {train_max:.4f}]")
    print(f"训练集归一化后范围: [{train_data.min():.4f}, {train_data.max():.4f}]")
    print(f"测试集归一化后范围: [{test_data.min():.4f}, {test_data.max():.4f}]")

    # ========== Step 4: 转换为图像 (ts_to_img) ==========
    
    # 转换为 torch tensor (已经是 (N, time_points, channels) 格式)
    train_data_tensor = torch.from_numpy(train_data).float()
    test_data_tensor = torch.from_numpy(test_data).float()
    
    # 获取序列长度
    seq_len = train_data.shape[1]  # 应该是 1000
    
    # 创建 DelayEmbedder
    if device == 'cuda' and not torch.cuda.is_available():
        print(f'  ⚠️  CUDA 不可用，使用 CPU')
        device = 'cpu'
    
    embedder = DelayEmbedder(
        device=device,
        seq_len=seq_len,
        delay=delay,
        embedding=embedding
    )
    
    # 转换训练集
    train_data_tensor = train_data_tensor.to(device)
    train_img = embedder.ts_to_img(train_data_tensor, pad=True, mask=0)
    
    # 转换测试集
    test_data_tensor = test_data_tensor.to(device)
    test_img = embedder.ts_to_img(test_data_tensor, pad=True, mask=0)
    
    # 转换标签为 tensor
    train_label_tensor = torch.from_numpy(train_label).long().to(device)
    test_label_tensor = torch.from_numpy(test_label).long().to(device)
    
    # ========== 完成 ==========
    print('=' * 80)
    print(f'  训练数据: {train_img.shape}, dtype={train_img.dtype}, device={train_img.device}')
    print(f'  训练标签: {train_label_tensor.shape}, dtype={train_label_tensor.dtype}')
    print(f'  测试数据: {test_img.shape}, dtype={test_img.dtype}, device={test_img.device}')
    print(f'  测试标签: {test_label_tensor.shape}, dtype={test_label_tensor.dtype}')
    print('=' * 80)
    
    return train_img, train_label_tensor, test_img, test_label_tensor
