import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..GoBoard import Board
from ..GoGame import GoGame
from ..GoNNet import GoNNet, GoNNetWrapper, net_config


def test_GoNNet_forward(board_size: int = 9) -> None:
    batch_size: int = net_config["batch_size"]
    game: GoGame = GoGame(n=board_size)
    nnet: GoNNet = GoNNet(game=game, args=net_config)
    state: Board = game.reset()
    policy, value = nnet.forward(
        s=torch.tensor(
            np.repeat(state.data, repeats=batch_size),
            dtype=torch.float,
        )
    )
    assert policy.shape == (batch_size, game.action_size())
    assert value.shape == (batch_size, 1)


def test_GoNNetWrapper_train(board_size: int = 9) -> None:
    batch_size: int = net_config["batch_size"]
    game: GoGame = GoGame(n=board_size)
    net_wrapper: GoNNetWrapper = GoNNetWrapper(game=game)
    board: np.ndarray = game.reset().data
    policy: np.ndarray = np.random.rand(game.action_size())
    value: np.ndarray = np.random.rand(1)
    net_wrapper.train(4 * batch_size * [(board, policy, value)])


def test_GoNNetWrapper_predict(board_size: int = 9) -> None:
    game: GoGame = GoGame(n=board_size)
    net_wrapper: GoNNetWrapper = GoNNetWrapper(game=game)
    policy, value = net_wrapper.predict(board=game.reset().data)
    assert policy.shape == (game.action_size(),)
    assert value.shape == (1,)
    assert np.all(0 <= policy) and np.all(policy <= 1)
    assert np.all(-1 <= value <= 1)


def test_GoNNetWrapper_save_checkpoint(board_size: int = 9) -> None:
    game: GoGame = GoGame(n=board_size)
    net_wrapper: GoNNetWrapper = GoNNetWrapper(game=game)
    folder: Path = Path(tempfile.mkdtemp())
    filename: str = "test_checkpoint.pth.tar"
    try:
        net_wrapper.save_checkpoint(folder=folder, filename=filename)
        assert (folder / filename).exists()
        net_wrapper.load_checkpoint(folder=folder, filename=filename)
    finally:
        shutil.rmtree(folder)
