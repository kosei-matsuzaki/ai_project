import torch.nn.init as init

class ComparativeTemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional=False):
        super(ComparativeTemporalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM層
        self.lstm = nn.LSTM(
            input_dim * 2,  # 基準動画と評価動画の特徴量を結合
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # 全結合層
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)

        # 初期化
        self._initialize_weights()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)  # Xavier初期化
            elif 'bias' in name:
                init.zeros_(param)  # バイアスはゼロに初期化

        # 全結合層
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)


class ComparativeTemporalCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, num_layers):
        super(ComparativeTemporalCNN, self).__init__()
        self.input_dim = input_dim * 2
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # Convolutional layers
        layers = []
        for i in range(num_layers):
            in_channels = self.input_dim if i == 0 else hidden_dim
            layers.append(
                nn.Conv1d(
                    in_channels,
                    hidden_dim,
                    kernel_size,
                    padding=kernel_size // 2  # Same padding
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
        self.conv_layers = nn.Sequential(*layers)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 初期化
        self._initialize_weights()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.conv_layers:
            if isinstance(m, nn.Conv1d):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He初期化
                if m.bias is not None:
                    init.zeros_(m.bias)

        # 全結合層
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
