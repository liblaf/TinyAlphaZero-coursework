import numpy as np
from TicTacToeGame import Board, TicTacToeGame


class AlphaBetaSearch:
    game: TicTacToeGame

    def __init__(self, game: TicTacToeGame) -> None:
        self.game = game

    def max_value(self, state: Board, alpha: float, beta: float) -> float:
        if self.game.is_terminal(state, 1) != 0:
            return self.game.is_terminal(state, 1)
        v = -np.inf
        valid_moves_mask = self.game.get_valid_moves(state, 1)
        valid_moves = np.where(valid_moves_mask == 1)[0]
        for a in valid_moves:
            v = max(v, self.min_value(self.game.next_state(state, 1, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, state: Board, alpha: float, beta: float) -> float:
        if self.game.is_terminal(state, 1) != 0:
            return self.game.is_terminal(state, 1)
        v = np.inf
        valid_moves_mask = self.game.get_valid_moves(state, -1)
        valid_moves = np.where(valid_moves_mask == 1)[0]
        for a in valid_moves:
            v = min(v, self.max_value(self.game.next_state(state, -1, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def get_best_move(self, state: Board, player: int) -> int:
        # convert player of current state to 1
        if player == -1:
            state = self.game.get_board(state, player)

        best_move, best_value = -1, -np.inf
        alpha, beta = -np.inf, np.inf
        valid_moves_mask = self.game.get_valid_moves(state, 1)
        valid_moves = np.where(valid_moves_mask == 1)[0]
        for a in valid_moves:
            v = self.min_value(self.game.next_state(state, 1, a), alpha, beta)
            if v > best_value:
                best_value = v
                best_move = a
        return best_move
