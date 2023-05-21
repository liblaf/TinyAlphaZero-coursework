from typing import Optional

import numpy as np
from AlphaBeta import AlphaBetaSearch
from MCTS import MCTS
from TicTacToeGame import Board, TicTacToeGame


class RandomPlayer:
    game: TicTacToeGame
    player: int

    def __init__(self, game: TicTacToeGame, player: int) -> None:
        self.game = game
        self.player = player

    def __str__(self) -> str:
        return "Random Player"

    def play(self, board: Board) -> int:
        valid_moves = self.game.get_valid_moves(board, self.player)
        a = np.random.choice(valid_moves.nonzero()[0])
        return a


class HumanTicTacToePlayer:
    game: TicTacToeGame
    player: int

    def __init__(self, game: TicTacToeGame, player: int) -> None:
        self.game = game
        self.player = player

    def __str__(self) -> str:
        return "Human Player"

    def play(self, board: Board) -> int:
        valid = self.game.get_valid_moves(board, self.player)
        while True:
            a = int(input())
            if valid[a]:
                break
            else:
                print("Invalid")
        return a


class AlphaBetaTicTacToePlayer:
    game: TicTacToeGame
    player: int
    policy: AlphaBetaSearch

    def __init__(self, game: TicTacToeGame, player: int) -> None:
        self.player = player
        self.game = game
        self.policy = AlphaBetaSearch(self.game)
        self.As = {}

    def __str__(self) -> str:
        return "AlphaBeta Player"

    def play(self, board: Board) -> int:
        s = self.game.get_string(board)
        if s not in self.As:
            self.As[s] = self.policy.get_best_move(board, self.player)
        return self.As[s]


class MCTSTicTacToePlayer:
    game: TicTacToeGame
    player: int
    policy: MCTS

    def __init__(
        self,
        game: TicTacToeGame,
        player: int,
        n_playout: int = 50,
        C: float = 1,
        param_file: Optional[str] = None,
    ) -> None:
        self.player = player
        self.game = game
        self.policy = MCTS(self.game, n_playout, C)
        if param_file is not None:
            self.policy.load_params(param_file)

    def __str__(self) -> str:
        return "MCTS Player"

    def eval(self) -> None:
        self.policy.eval()

    def train(self) -> None:
        self.policy.train()

    def play(self, board: Board) -> int:
        return np.argmax(
            self.policy.get_action_prob(board, self.player)
            * self.game.get_valid_moves(board, self.player)
        )

    def save_params(self, file_name: str) -> None:
        self.policy.save_params(file_name)

    def load_params(self, file_name: str) -> None:
        self.policy.load_params(file_name)
