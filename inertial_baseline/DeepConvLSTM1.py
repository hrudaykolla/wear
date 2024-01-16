# ------------------------------------------------------------------------
# DeepConvLSTM model based on architecture suggested by Ordonez and Roggen 
# https://www.mdpi.com/1424-8220/16/1/115
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

from torch import nn


class DeepConvLSTM1(nn.Module):
    def __init__(self, channels, classes, window_size, conv_kernels=64, conv_kernel_size=5, lstm_units=128, lstm_layers=2, dropout=0.5, feature_extract=None):
        super(DeepConvLSTM1, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.bn1 = nn.BatchNorm2d(conv_kernels)
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.bn2 = nn.BatchNorm2d(conv_kernels)
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.bn3 = nn.BatchNorm2d(conv_kernels)
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.bn4 = nn.BatchNorm2d(conv_kernels)
        self.lstm = nn.LSTM(channels * conv_kernels, lstm_units, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()
        self.final_seq_len = window_size - (conv_kernel_size - 1) * 4
        self.lstm_units = lstm_units
        self.classes = classes
        self.feature_extract = feature_extract

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        if self.feature_extract == 'conv':
            return x.view(x.shape[0], -1)
        x = x.permute(0, 2, 1, 3)

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, h = self.lstm(x)
        x = x[:,-1,:]
        if self.feature_extract == 'lstm':
            return x
        x = self.dropout(x)
        x = self.classifier(x)
        return x