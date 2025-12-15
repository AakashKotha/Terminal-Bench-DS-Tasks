import numpy as np

class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.player = 1 # 1 or -1
        self.last_move = None
        self.winner = None

    def get_valid_moves(self):
        if self.winner is not None:
            return []
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def step(self, col):
        if col not in self.get_valid_moves():
            raise ValueError(f"Invalid move {col}")
        
        # Drop piece
        row = np.max(np.where(self.board[:, col] == 0))
        self.board[row, col] = self.player
        self.last_move = (row, col)
        
        if self.check_win(row, col):
            self.winner = self.player
        
        self.player *= -1 # Switch turn
        return self

    def check_win(self, r, c):
        color = self.board[r, c]
        # Directions: Horizontal, Vertical, Diagonal /, Diagonal \
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # Positive dir
            for i in range(1, 4):
                nr, nc = r + dr*i, c + dc*i
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr, nc] == color:
                    count += 1
                else:
                    break
            # Negative dir
            for i in range(1, 4):
                nr, nc = r - dr*i, c - dc*i
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr, nc] == color:
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False

    def is_terminal(self):
        return self.winner is not None or len(self.get_valid_moves()) == 0

    def get_result(self, perspective_player):
        """
        Returns +1 if perspective_player won, -1 if lost, 0 if draw.
        """
        if self.winner == perspective_player:
            return 1.0
        elif self.winner == -perspective_player:
            return -1.0
        else:
            return 0.0 # Draw or running

    def copy(self):
        new_game = Connect4()
        new_game.board = self.board.copy()
        new_game.player = self.player
        new_game.winner = self.winner
        return new_game