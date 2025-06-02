from sample_ai import ChessAI
import torch
import numpy as np
from model import AlphaZeroChessNet

class AlphaZeroAI(ChessAI):
    def __init__(self, model_path, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.model = AlphaZeroChessNet()
        
        # Xử lý tương thích với cả checkpoint và model state dict
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Nếu là checkpoint từ quá trình training
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Nếu chỉ là state dict
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with an untrained model")
        
        self.model.eval()

    def board_to_tensor(self, board, color):
        # Giả sử board là một object có thể truy cập trạng thái từng ô
        # Encode đơn giản: 0 = empty, 1-6 = white, 7-12 = black
        # Bạn cần thay thế bằng encode thực tế nếu có
        state = np.zeros((14, 8, 8), dtype=np.float32)
        piece_map = board.get_piece_map()  # {(row, col): piece}
        for (row, col), piece in piece_map.items():
            idx = self.piece_to_channel(piece, color)
            if idx is not None:
                state[idx, row, col] = 1.0
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

    def piece_to_channel(self, piece, color):
        # Đảm bảo piece.type là int (nếu là enum hoặc str thì chuyển về int)
        # Nếu piece.type là str, cần ánh xạ về số nguyên
        type_map = {
            "PAWN": 0, "KNIGHT": 1, "BISHOP": 2, "ROOK": 3, "QUEEN": 4, "KING": 5,
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5  # Nếu đã là số
        }
        color_map = {
            "WHITE": 0, "BLACK": 1, 0: 0, 1: 1
        }
        # Xử lý piece_type
        piece_type = piece.type
        if isinstance(piece_type, str):
            piece_type = type_map.get(piece_type, -1)
        elif hasattr(piece_type, "value"):
            piece_type = piece_type.value
        # Xử lý piece_color
        piece_color = piece.color
        if isinstance(piece_color, str):
            is_white = color_map.get(piece_color, -1) == 0
        elif hasattr(piece_color, "value"):
            is_white = piece_color.value == 0
        else:
            is_white = piece_color == 0
        if not isinstance(piece_type, int) or piece_type < 0 or piece_type > 5:
            return None
        return piece_type + (0 if is_white else 6)
    
    def legal_moves(self, board, color):
        # Trả về danh sách các nước đi hợp lệ [(from_pos, to_pos), ...]
        # Giả sử board có hàm get_legal_moves(color) trả về [(from_row, from_col, to_row, to_col), ...]
        return board.get_legal_moves(color)

    def move_to_index(self, move):
        # Kiểm tra cấu trúc của move
        # Giả sử move là tuple (from_row, from_col, to_row, to_col)
        if len(move) != 4:
            # Fallback nếu move không đúng định dạng
            return 0
        from_row, from_col, to_row, to_col = move
        return from_row * 8 * 8 * 8 + from_col * 8 * 8 + to_row * 8 + to_col

    def index_to_move(self, index, board, color):
        # Decode index về move
        from_row = index // (8*8*8)
        from_col = (index // (8*8)) % 8
        to_row = (index // 8) % 8
        to_col = index % 8
        move = (from_row, from_col, to_row, to_col)
        legal = self.legal_moves(board, color)
        if move in legal:
            return move
        # Nếu move không hợp lệ, chọn nước đầu tiên hợp lệ
        return legal[0]

    def mcts(self, board, color, simulations=50):
        # Đơn giản: chỉ dùng policy head để chọn nước đi xác suất cao nhất trong các nước hợp lệ
        state_tensor = self.board_to_tensor(board, color)
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        legal = self.legal_moves(board, color)
        
        # Kiểm tra nếu không có nước đi hợp lệ
        if not legal:
            return None
            
        move_indices = [self.move_to_index(mv) for mv in legal]
        legal_probs = []
        
        # Đảm bảo chỉ lấy các index nằm trong giới hạn của policy vector
        policy_size = len(policy)
        for idx in move_indices:
            if idx < policy_size:
                legal_probs.append((policy[idx], idx))
            else:
                legal_probs.append((0.0, idx))
                
        if not legal_probs:
            return legal[0]
            
        legal_probs.sort(reverse=True)
        best_idx = legal_probs[0][1]
        return self.index_to_move(best_idx, board, color)
    
    def get_move(self, board, color):
        """
        Get the next move from the AlphaZero AI.
        
        Args:
            board: The current chess board state
            color: The color (PieceColor.WHITE or PieceColor.BLACK) the AI is playing as
            
        Returns:
            Position objects for from_position and to_position (or None if no move possible)
        """
        move = self.mcts(board, color)
        if move is None:
            return None
            
        # Convert tuple format to Position objects if needed
        from_row, from_col, to_row, to_col = move
        from chess_board import Position  # Đảm bảo import Position
        
        from_position = Position(from_row, from_col)
        to_position = Position(to_row, to_col)
        
        return from_position, to_position