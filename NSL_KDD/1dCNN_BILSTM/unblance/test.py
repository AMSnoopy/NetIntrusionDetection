from data_process import preprocess_data
from tqdm import tqdm
from one_dCNN_BILSTM import CNNLSTMModel
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 判断是否使用GPU
train_loader, test_loader=preprocess_data()#数据预处理
# 初始化模型和优化器
input_dim = 163
cnn_out_channels = 64
lstm_hidden_dim = 128
output_dim = 5  # 假设是七分类任务
dropout_prob = 0.2
model = CNNLSTMModel(input_dim, cnn_out_channels, lstm_hidden_dim, output_dim,dropout_prob)

# 加载train.py里训练好的模型
model.load_state_dict(torch.load('Mould/best_model.pth'))
# 将模型设置为评估模式。在评估模式下，模型会关闭 Dropout 等仅在训练过程中使用的特性，
# 并且会使用已经训练好的参数进行预测。
model.eval()
# 在模型评估部分添加以下代码
classes =['Normal', 'Dos', 'Probe', 'R2L', 'U2R']
all_preds = []
all_labels = []

with torch.no_grad():  # 在预测时不计算梯度，节省内存
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到模型所在的设备上
        outputs = model.forward(inputs)  # 模型预测
        _, preds = torch.max(outputs, 1)  # 获取预测类别，_表示不关心最大值的具体数值

        all_preds.extend(preds.cpu().numpy())  # 收集预测结果
        all_labels.extend(labels.cpu().numpy())  # 收集真实标签
# 转换为numpy数组以便于计算
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
from collections import Counter
def print_count_numbers_separately(arr1, arr2):
    # 使用Counter计算每个数组中元素的频率
    counts1 = Counter(arr1)
    counts2 = Counter(arr2)

    # 打印数组1的计数
    print("Array 1:")
    for num, freq in counts1.items():
        print(f"Number {num}: {freq} times")

    # 打印数组2的计数
    print("\nArray 2:")
    for num, freq in counts2.items():
        print(f"Number {num}: {freq} times")


overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Overall Accuracy: {overall_accuracy}")

# 计算每个类别的准确率、精确率、召回率、F1分数和支持度
precision, recall, f1_score, support = precision_recall_fscore_support(all_labels, all_preds, average=None)

# 输出结果
print("Class-wise Metrics:")
for i in range(5):  # 假设是七分类问题
    print(f"class {classes[i]}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1 Score={f1_score[i]:.4f}, "
          f"Support={support[i]}")

# 将非零值视为正类（其他值视为负类）
binary_preds = (all_preds != 0).astype(int)  # 非0为为ture转转化则为1
binary_labels = (all_labels != 0).astype(int)

# 计算二分类准确率
binary_accuracy = accuracy_score(binary_labels, binary_preds)
print(f"binary Accuracy: {binary_accuracy}")

# 计算二分类的指标
report = classification_report(binary_labels, binary_preds, zero_division=0)
print("Binary Classification Report:\n", report)
print_count_numbers_separately(all_labels, all_preds)