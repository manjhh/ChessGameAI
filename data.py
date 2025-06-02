import pickle
import numpy as np
import os

num_samples = 1000
in_channels = 14
board_size = 8
n_moves = 4672

data = []
for _ in range(num_samples):
    # Giả lập trạng thái bàn cờ
    state = np.random.randint(0, 2, size=(in_channels, board_size, board_size)).astype(np.float32)
    # Giả lập xác suất MCTS: chọn ngẫu nhiên một số nước đi hợp lệ, phân phối xác suất cho chúng
    legal_moves = np.random.choice(n_moves, size=np.random.randint(10, 40), replace=False)
    policy = np.zeros(n_moves, dtype=np.float32)
    probs = np.random.dirichlet(np.ones(len(legal_moves)))
    policy[legal_moves] = probs
    # Giá trị kết quả trận đấu
    value = np.random.choice([-1.0, 0.0, 1.0])
    data.append((state, policy, value))

os.makedirs("data", exist_ok=True)
with open("data/train_data.pkl", "wb") as f:
    pickle.dump(data, f)