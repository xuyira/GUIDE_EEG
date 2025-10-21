from data_preprocess import eegdata_preprocess

train_data, train_label, test_data, test_label = eegdata_preprocess(
    testsubj=1,        # 测试受试者编号 (1-9)
    dataset='bci2a',   # 数据集
    delay=15,          # 时间延迟参数
    embedding=64,      # 嵌入维度
    device='cuda'      # 计算设备
)

