import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroChessNet(nn.Module):
    def __init__(self, board_size=8, num_residual_blocks=5, filters=128):
        super(AlphaZeroChessNet, self).__init__()
        self.board_size = board_size
        self.conv_init = nn.Conv2d(in_channels=14,  # ví dụ: encode 7 loại quân 2 màu 
                                   out_channels=filters, 
                                   kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(filters)
        
        # Các khối residual
        self.res_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.res_blocks.append(self._build_residual_block(filters))
        
        # Phần đầu ra policy
        self.conv_policy = nn.Conv2d(filters, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * board_size * board_size, board_size * board_size * 73) 
        # 73 nước đi tối đa có thể tùy cách mã hóa. Cần điều chỉnh cho hợp lý.

        # Phần đầu ra value
        self.conv_value = nn.Conv2d(filters, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(board_size * board_size, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def _build_residual_block(self, filters):
        return nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
        )

    def forward(self, x):
        # Khối feature extraction
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks:
            residual = x
            out = block(x)
            x = F.relu(residual + out)

        # Đầu ra policy
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p)
        # Đầu ra ma trận xác suất cho các nước đi tiềm năng
        policy = F.log_softmax(p, dim=1)

        # Đầu ra value
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        value = torch.tanh(self.fc_value2(v))

        return policy, value