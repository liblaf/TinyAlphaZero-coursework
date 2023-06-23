import numpy as np

from .. import GoNNetWrapper
from ..GoBoard import Board
from ..GoGame import GoGame
from ..MCTS import MCTS


def test_MCTS_train(board_size: int = 5) -> None:
    game: GoGame = GoGame(n=board_size)
    nnet: GoNNetWrapper = GoNNetWrapper(game=game)
    mcts: MCTS = MCTS(game=game, nnet=nnet, num_sims=100, C=1.0)
    mcts.train()
    assert mcts.training == True


def test_MCTS_eval(board_size: int = 5) -> None:
    game: GoGame = GoGame(n=board_size)
    nnet: GoNNetWrapper = GoNNetWrapper(game=game)
    mcts: MCTS = MCTS(game=game, nnet=nnet, num_sims=100, C=1.0)
    mcts.eval()
    assert mcts.training == False


def test_MCTS_get_action_prob(board_size: int = 5) -> None:
    game: GoGame = GoGame(n=board_size)
    nnet: GoNNetWrapper = GoNNetWrapper(game=game)
    mcts: MCTS = MCTS(game=game, nnet=nnet, num_sims=100, C=1.0)
    board: Board = game.reset()
    action_prob: np.ndarray = mcts.get_action_prob(board=board, player=1)
    assert action_prob.shape == (game.action_size(),)
    assert np.all(0 <= action_prob) and np.all(action_prob <= 1)
    np.testing.assert_almost_equal(np.sum(action_prob), 1.0)
