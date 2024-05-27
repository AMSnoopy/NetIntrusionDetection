import torch
import torch.nn as nn


# 定义自注意力层
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = torch.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, output_dim, dropout_prob):
        super(CNNLSTMModel, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.dropout1 = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.dropout2 = nn.Dropout(dropout_prob)
        self.bn2 = nn.BatchNorm1d(128)

        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.dropout3 = nn.Dropout(dropout_prob)
        self.bn3 = nn.BatchNorm1d(32)
        self.lstm = nn.LSTM(32, lstm_hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout_prob)
        self.bn_lstm = nn.BatchNorm1d(lstm_hidden_dim * 2)
        self.attention_lstm = SelfAttention(lstm_hidden_dim * 2)
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_out1 = self.cnn1(x)
        # print(cnn_out1.shape)
        cnn_out1 = self.dropout1(cnn_out1)
        cnn_out1 = self.bn1(cnn_out1)
        # print(cnn_out1.shape)
        cnn_out2 = self.cnn2(cnn_out1)
        cnn_out2 = self.dropout2(cnn_out2)
        cnn_out2 = self.bn2(cnn_out2)
        # print(cnn_out2.shape)
        cnn_out3 = self.cnn3(cnn_out2)
        cnn_out3 = self.dropout3(cnn_out3)
        cnn_out3 = self.bn3(cnn_out3)
        # print(cnn_out3.shape)  # [64,32,24]
        lstm_out, _ = self.lstm(cnn_out3)
        #print(lstm_out.shape)#torch.Size([64, 32, 256])
        lstm_out = self.dropout_lstm(lstm_out)
       #print(lstm_out.shape)#torch.Size([64, 256, 32])
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.bn_lstm(lstm_out)
        # print(lstm_out.shape)torch.Size([64, 32, 256])
        lstm_out = lstm_out.permute(0, 2, 1)
        attention_out, _ = self.attention_lstm(lstm_out)
        # print(lstm_out.shape) torch.Size([64, 10])

        output = self.fc(attention_out)
       # print(output.shape)
        return output
