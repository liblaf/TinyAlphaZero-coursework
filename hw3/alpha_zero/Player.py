from abc import ABC
from pathlib import Path
from typing import Optional, Union

import numpy as np
from typing_extensions import Self

from . import GoNNetWrapper
from .GoBoard import Board
from .GoGame import GoGame
from .MCTS import MCTS


class Player(ABC):
    player: int = 0

    def __str__(self) -> str:
        ...

    def play(self, board: Board, player: Optional[int] = None) -> int:
        ...

    def copy(self) -> Self:
        ...


class RandomPlayer(Player):
    game: GoGame
    player: int = 0

    def __init__(self, game: GoGame, player: int) -> None:
        self.game = game
        self.player = player

    def __str__(self) -> str:
        return "Random Player"

    def play(self, board: Board, player: Optional[int] = None) -> int:
        valid_moves: np.ndarray = self.game.get_valid_moves(
            board, self.player if player is None else player
        )
        a = np.random.choice(valid_moves.nonzero()[0])
        return a

    def copy(self) -> Self:
        return RandomPlayer(self.game, self.player)


class AlphaZeroPlayer(Player):
    game: GoGame
    player: int = 0
    nnet: GoNNetWrapper
    checkpoint_path: Union[str, Path]
    mcts: MCTS

    def __init__(
        self, game: GoGame, checkpoint_path: Union[str, Path], num_sims: int, C: float
    ) -> None:
        self.game = game
        self.nnet = GoNNetWrapper(game)
        self.checkpoint_path = checkpoint_path
        if Path(checkpoint_path).exists():
            self.nnet.load_checkpoint(checkpoint_path)
        self.mcts = MCTS(game, self.nnet, num_sims, C)

    def __str__(self) -> str:
        return f"AlphaZero Player({self.checkpoint_path})"

    def play(self, board: Board, player: Optional[int] = None) -> int:
        return int(
            np.argmax(
                self.mcts.get_action_prob(
                    board, player=self.player if player is None else player
                )
            )
        )

    def copy(self) -> Self:
        new_player: AlphaZeroPlayer = AlphaZeroPlayer(
            game=self.game,
            checkpoint_path=self.checkpoint_path,
            num_sims=self.mcts.num_sims,
            C=self.mcts.C,
        )
        new_player.player = self.player
        new_player.nnet = self.nnet
        new_player.mcts = self.mcts
        return new_player


class FastEvalPlayer(Player):
    mcts: MCTS
    player: int = 0

    def __init__(self, mcts: MCTS) -> None:
        self.mcts = mcts
        self.player = 0

    def play(self, board: Board, player: Optional[int] = None) -> int:
        self.mcts.eval()
        action_probs: np.ndarray = self.mcts.get_action_prob(
            board, player=self.player if player is None else player
        )
        return np.random.choice(len(action_probs), p=action_probs)

    def copy(self) -> Self:
        new_player: FastEvalPlayer = FastEvalPlayer(mcts=self.mcts)
        new_player.player = self.player
        return new_player
