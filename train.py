import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from config import get_config
from model import AlphaZeroChessNet
from data_utils import ChessDataset

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, policy_losses, value_losses = 0.0, 0.0, 0.0
    total_examples = 0
    
    for state_batch, policy_batch, value_batch in loader:
        state_batch = state_batch.to(device)
        policy_batch = policy_batch.to(device)
        value_batch = value_batch.to(device)

        # Forward pass
        policy_pred, value_pred = model(state_batch)
        
        # Dùng KL divergence cho policy loss thay vì nll_loss
        # Log softmax được áp dụng để thành phân phối xác suất
        policy_pred_prob = torch.log_softmax(policy_pred, dim=1)
        policy_loss = torch.nn.functional.kl_div(
            policy_pred_prob, 
            policy_batch, 
            reduction='batchmean',
            log_target=False
        )
        
        # MSE loss cho value prediction
        value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), value_batch)

        # Tổng hợp loss
        loss = policy_loss + value_loss

        # Backward pass và update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tính tổng loss
        batch_size = state_batch.size(0)
        total_loss += loss.item() * batch_size
        policy_losses += policy_loss.item() * batch_size
        value_losses += value_loss.item() * batch_size
        total_examples += batch_size

    return {
        'total_loss': total_loss / total_examples,
        'policy_loss': policy_losses / total_examples,
        'value_loss': value_losses / total_examples
    }

def main():
    args = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tạo thư mục lưu model nếu chưa tồn tại
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Tải dữ liệu
    dataset = ChessDataset(args.train_data_path)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=4
    )
    print(f"Loaded {len(dataset)} training examples")
    
    # Khởi tạo model
    model = AlphaZeroChessNet(
        num_residual_blocks=args.num_residual_blocks,
        filters=args.filters
    ).to(device)
    
    # Khởi tạo tham số từ checkpoint nếu có
    start_epoch = 0
    if os.path.exists(args.model_path):
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                print(f"Loaded checkpoint from epoch {start_epoch-1}")
            else:
                # Trường hợp checkpoint chỉ lưu model state dict
                model.load_state_dict(checkpoint)
                print("Loaded model weights")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Scheduler để giảm learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"Starting training for {args.epochs} epochs")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train một epoch
        losses = train_one_epoch(model, loader, optimizer, device)
        
        # Scheduler step
        scheduler.step(losses['total_loss'])
        
        # Log kết quả
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {losses['total_loss']:.4f} | "
              f"Policy Loss: {losses['policy_loss']:.4f} | "
              f"Value Loss: {losses['value_loss']:.4f} | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Lưu checkpoint, bao gồm model state và metadata
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": losses['total_loss'],
        }
        
        # Lưu checkpoint
        torch.save(checkpoint, args.model_path)
        
        # Lưu model tốt nhất
        if losses['total_loss'] < best_loss:
            best_loss = losses['total_loss']
            torch.save(checkpoint, args.model_path.replace('.pth', '_best.pth'))
            print(f"New best model saved with loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()