from abc import ABC
from pathlib import Path
from typing import Union

import numpy as np

from . import GoNNetWrapper
from .GoBoard import Board
from .GoGame import GoGame
from .MCTS import MCTS


class Player(ABC):
    player: int = 0

    def __str__(self) -> str:
        ...

    def play(self, board: Board) -> int:
        ...


class RandomPlayer(Player):
    game: GoGame
    player: int = 0

    def __init__(self, game: GoGame, player: int) -> None:
        self.game = game
        self.player = player

    def __str__(self) -> str:
        return "Random Player"

    def play(self, board: Board) -> int:
        valid_moves: np.ndarray = self.game.get_valid_moves(board, self.player)
        a = np.random.choice(valid_moves.nonzero()[0])
        return a


class AlphaZeroPlayer(Player):
    game: GoGame
    player: int = 0
    nnet: GoNNetWrapper
    checkpoint_path: Union[str, Path]
    mcts: MCTS

    def __init__(
        self, game: GoGame, checkpoint_path: Union[str, Path], num_sims: int, C: float
    ) -> None:
        self.nnet = GoNNetWrapper(game)
        self.checkpoint_path = checkpoint_path
        # self.nnet.load_checkpoint(checkpoint_path)
        self.mcts = MCTS(game, self.nnet, num_sims, C)

    def __str__(self) -> str:
        return f"AlphaZero Player({self.checkpoint_path})"

    def play(self, board: Board) -> int:
        return int(np.argmax(self.mcts.get_action_prob(board, player=self.player)))


class FastEvalPlayer(Player):
    mcts: MCTS
    player: int = 0

    def __init__(self, mcts: MCTS) -> None:
        self.mcts = mcts
        self.player = 0

    def play(self, board: Board) -> int:
        return int(np.argmax(self.mcts.get_action_prob(board, self.player)))
