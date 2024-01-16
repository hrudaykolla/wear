# ------------------------------------------------------------------------
# DeepConvLSTM model based on architecture suggested by Ordonez and Roggen 
# https://www.mdpi.com/1424-8220/16/1/115
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

from torch import nn


class DeepConvLSTMwithAttention(nn.Module):
    def __init__(self, channels, classes, window_size, conv_kernels=64, conv_kernel_size=5, lstm_units=128, lstm_layers=2, dropout=0.5, feature_extract=None):
        super(DeepConvLSTMwithAttention, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.lstm = nn.LSTM(channels * conv_kernels, lstm_units, num_layers=lstm_layers, batch_first=True)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=lstm_units, num_heads=2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()
        self.final_seq_len = window_size - (conv_kernel_size - 1) * 4
        self.lstm_units = lstm_units
        self.classes = classes
        self.feature_extract = feature_extract

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        if self.feature_extract == 'conv':
            return x.view(x.shape[0], -1)
        x = x.permute(0, 2, 1, 3)

        x = x.reshape(x.shape[0], x.shape[1], -1)

        x, h = self.lstm(x)

        if self.feature_extract == 'lstm':
            return x[:,-1,:]
        
        x = x.permute(1, 0, 2)  # Swap batch and sequence length dimensions for attention layer
        x, _ = self.multihead_attention(x, x, x)  # Self-attention
        
        x = x.permute(1, 0, 2)  # Swap back to batch-first format

        x = x[:,-1,:]  # Take the last time step output
        x = self.dropout(x)
        x = self.classifier(x)
        return x
