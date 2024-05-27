import torch
from data_process import preprocess_data
from tqdm import tqdm
from one_dCNN_BILSTM import CNNLSTMModel
import torch.nn as nn
import torch.optim as optim

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
train_loader, test_loader = preprocess_data()

# 初始化模型和优化器
input_dim = 163
cnn_out_channels = 64
lstm_hidden_dim = 128
output_dim = 5  # 假设是七分类任务
dropout_prob = 0.1  # 任务越大，丢弃率要设置得更大，否则容易过拟合，最好在0.2-0.5之间
model = CNNLSTMModel(input_dim, cnn_out_channels, lstm_hidden_dim, output_dim, dropout_prob).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15
best_accuracy = 0  # 用于记录最佳准确率

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0

    # 使用tqdm显示训练进度
    for batch_idx, (X_batch, y_batch) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)):
        optimizer.zero_grad()
        # 将数据移动到当前设备
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss = loss.to(device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == y_batch).sum().item()

    # 计算平均损失和训练准确率
    avg_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / len(train_loader.dataset)

    # 在每个epoch结束时，评估并在测试集上计算准确率
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X_test, y_test in test_loader:
            # 将测试数据移动到当前设备
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

        test_accuracy = correct / total

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # 如果当前测试集准确率优于之前的最佳准确率，则保存模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), 'Mould/best_model.pth')
        print(f'Saved best model with test accuracy: {best_accuracy:.4f}')
