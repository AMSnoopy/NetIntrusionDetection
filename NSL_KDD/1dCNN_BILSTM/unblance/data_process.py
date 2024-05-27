from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
def preprocess_data():
    # 加载和预处理数据
    data = pd.read_csv('E:\\code\\python\\Paper_code\\数据集\\NSL_KDD_Dataset\\merged_unguiyi.csv')

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    # # 训练和测试数据划分
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 划分训练集和测试集，按照 y 的类别进行分层抽样
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 初始化scaler并用训练集拟合
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    # 对训练集和测试集进行归一化
    X_train = scaler.transform(X_train)
    X_test= scaler.transform(X_test)

    # 将数据转换为 PyTorch 的 Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_loader, test_loader