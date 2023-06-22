from typing import List, Tuple

import numpy as np

from ..GoBoard import Board, Stone
from ..GoGame import GoGame


def test_reset() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    assert np.all(board.data == Stone.EMPTY)


def test_obs_size() -> None:
    game: GoGame = GoGame(n=5)
    assert game.obs_size() == (5, 5)


def test_action_size() -> None:
    game: GoGame = GoGame(n=5)
    assert game.action_size() == 26


def test_get_board_self() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    board.data.fill(Stone.BLACK)
    assert np.all(game.get_board(board=board, player=1).data == Stone.BLACK)


def test_get_board_opponent() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    board.data.fill(Stone.BLACK)
    assert np.all(game.get_board(board=board, player=-1).data == Stone.WHITE)


def test_next_state() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    board: Board = game.next_state(board=board, player=1, action=0)
    assert np.all(
        board.to_numpy()
        == np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
    )


def test_next_state_pass() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    board: Board = game.next_state(board=board, player=1, action=25)
    assert np.all(board.data == Stone.EMPTY)


def test_valid_moves() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    valid_moves: np.ndarray = game.get_valid_moves(board=board, player=1)
    assert valid_moves.shape == (26,)
    assert np.all(valid_moves)


def test_terminal_last_move_is_pass() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    board.last_move_is_pass = (True, True)
    assert game.is_terminal(board=board, player=1) == 1e-4
    assert game.is_terminal(board=board, player=-1) == 1e-4


def test_terminal_too_many_moves() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    board.num_moves = board.n * board.n * 4 + 1
    assert game.is_terminal(board=board, player=1) == 1e-4
    assert game.is_terminal(board=board, player=-1) == 1e-4


def test_terminal_black_wins() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    board.data.fill(Stone.BLACK)
    board.last_move_is_pass = (True, True)
    assert game.is_terminal(board=board, player=1) == 1
    assert game.is_terminal(board=board, player=-1) == -1


def test_terminal_white_wins() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    board.data.fill(Stone.WHITE)
    board.last_move_is_pass = (True, True)
    assert game.is_terminal(board=board, player=1) == -1
    assert game.is_terminal(board=board, player=-1) == 1


def test_terminal_not_over() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    assert game.is_terminal(board=board, player=1) == 0
    assert game.is_terminal(board=board, player=-1) == 0


def test_get_transform_data() -> None:
    game: GoGame = GoGame(n=5)
    board: Board = game.reset()
    policy: np.ndarray = np.zeros(shape=(game.action_size()))
    data: List[Tuple[Board, np.ndarray]] = game.get_transform_data(
        board=board, policy=policy
    )
    assert len(data) == 8
    assert all([len(p) == game.action_size() for _, p in data])
