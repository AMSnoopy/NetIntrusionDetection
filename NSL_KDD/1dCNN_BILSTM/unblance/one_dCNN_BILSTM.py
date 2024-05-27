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


# 定义模型
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, output_dim, dropout_prob):
        super(CNNLSTMModel, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.attention1 = SelfAttention(64)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.attention2 = SelfAttention(128)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.attention3 = SelfAttention(32)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.lstm = nn.LSTM(32, lstm_hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.attention_lstm = SelfAttention(lstm_hidden_dim * 2)
        self.dropout_lstm = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        cnn_out1 = self.cnn1(x)
        attn_out1, _ = self.attention1(cnn_out1.transpose(1, 2))
        attn_out1 = self.dropout1(attn_out1)

        cnn_out2 = self.cnn2(cnn_out1)
        attn_out2, _ = self.attention2(cnn_out2.transpose(1, 2))
        attn_out2 = self.dropout2(attn_out2)

        cnn_out3 = self.cnn3(cnn_out2)
        attn_out3, _ = self.attention3(cnn_out3.transpose(1, 2))
        attn_out3 = self.dropout3(attn_out3)

        lstm_out, _ = self.lstm(attn_out3.unsqueeze(1))
        attention_out, _ = self.attention_lstm(lstm_out)
        attention_out = self.dropout_lstm(attention_out)

        output = self.fc(attention_out)
        return output
