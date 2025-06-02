import argparse

def get_config():
    parser = argparse.ArgumentParser(description="AlphaZero Chess Training Config")
    parser.add_argument("--epochs", type=int, default=100, help="Số vòng huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_residual_blocks", type=int, default=5, help="Số residual blocks trong mạng")
    parser.add_argument("--filters", type=int, default=128, help="Số kênh feature maps")
    parser.add_argument("--model_path", type=str, default="checkpoints/model.pth", help="Đường dẫn lưu mô hình")
    parser.add_argument("--train_data_path", type=str, default="data/train_data.pkl", help="Đường dẫn dữ liệu huấn luyện")
    args = parser.parse_args()
    return args