from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import numpy as np

from .. import GoNNetWrapper
from ..GoBoard import Board
from ..GoGame import GoGame
from ..main import args
from ..train_alphazero import Trainer, static_collect_single_game


def test_train_alphazero_collect_single_game(board_size: int = 9) -> None:
    game: GoGame = GoGame(n=board_size)
    trainer: Trainer = Trainer(game=game, nnet=GoNNetWrapper(game=game), config=args)
    data_pack: List[Tuple[Board, np.ndarray, float]] = trainer.collect_single_game()
    board, policy, value = data_pack[0]
    assert board.data.shape == (board_size, board_size)
    assert policy.shape == (game.action_size(),)
    assert np.all(0 <= policy) and np.all(policy <= 1)
    assert -1 <= value <= 1


def test_train_alphazero_static_collect_single_game(board_size: int = 9) -> None:
    game: GoGame = GoGame(n=board_size)
    trainer: Trainer = Trainer(game=game, nnet=GoNNetWrapper(game=game), config=args)
    data_pack: List[Tuple[Board, np.ndarray, float]] = static_collect_single_game(
        board_size=board_size,
        nnet=trainer.next_net,
        num_sims=trainer.config["num_sims"],
        cpuct=trainer.config["cpuct"],
    )
    board, policy, value = data_pack[0]
    assert board.data.shape == (board_size, board_size)
    assert policy.shape == (game.action_size(),)
    assert np.all(0 <= policy) and np.all(policy <= 1)
    assert -1 <= value <= 1
