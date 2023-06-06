import numpy as np

from ..GoBoard import Board


def test_capture_0() -> None:
    board: Board = Board(n=5)
    board.load_from_numpy(
        np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, 1, -1, 0],
                [-1, 1, 0, 1, -1],
                [0, -1, 1, -1, 0],
                [0, 0, -1, 0, 0],
            ]
        )
    )
    # print(board.__str__())
    assert set(board.valid_moves(1)) == set(
        [
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 4),
            (3, 0),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
        ]
    )
    board.add_stone(2, 2, -1)
    # print(board.__str__())
    assert (
        board.to_numpy()
        == np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, 0, -1, 0],
                [-1, 0, -1, 0, -1],
                [0, -1, 0, -1, 0],
                [0, 0, -1, 0, 0],
            ]
        )
    ).all()
    assert set(board.valid_moves(1)) == set(
        [
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 4),
            (3, 0),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
        ]
    )
    assert set(board.valid_moves(-1)) == set(
        [
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 2),
            (1, 4),
            (2, 1),
            (2, 3),
            (3, 0),
            (3, 2),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
        ]
    )


def test_capture_1() -> None:
    board: Board = Board(n=5)
    board.load_from_numpy(
        np.array(
            [
                [0, -1, 1, -1, 0],
                [-1, 1, 1, 1, -1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, -1, 0],
            ]
        )
    )
    # print(board.__str__())
    assert set(board.valid_moves(1)) == set([(0, 0), (0, 4), (3, 4), (4, 4)])
    assert set(board.valid_moves(-1)) == set([(3, 4), (4, 4)])
    board.add_stone(3, 4, -1)
    # print(board.__str__())
    assert (
        board.to_numpy()
        == np.array(
            [
                [0, -1, 0, -1, 0],
                [-1, 0, 0, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
                [0, 0, 0, -1, 0],
            ]
        )
    ).all()


def test_ko_rule() -> None:
    board: Board = Board(n=5)
    board.load_from_numpy(
        np.array(
            [
                [0, -1, 1, 0, 0],
                [-1, 0, -1, 1, 0],
                [0, -1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        )
    )
    assert set(board.valid_moves(-1)) == set(
        [
            (0, 0),
            (0, 3),
            (0, 4),
            (1, 1),
            (1, 4),
            (2, 0),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
        ]
    )
    assert set(board.valid_moves(1)) == set(
        [
            (0, 3),
            (0, 4),
            (1, 1),
            (1, 4),
            (2, 0),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
        ]
    )
    # print(board.__str__())
    board.add_stone(1, 1, 1)
    # print(board.__str__())
    assert (
        board.to_numpy()
        == np.array(
            [
                [0, -1, 1, 0, 0],
                [-1, 1, 0, 1, 0],
                [0, -1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        )
    ).all()
    assert set(board.valid_moves(-1)) == set(
        [
            (0, 0),
            (0, 3),
            (0, 4),
            (1, 4),
            (2, 0),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
        ]
    )
    assert set(board.valid_moves(1)) == set(board.valid_moves(-1))
    board.add_stone(1, 4, -1)
    board.add_stone(2, 3, 1)
    # print(board.__str__())
    assert set(board.valid_moves(-1)) == set(
        [
            (0, 0),
            (0, 3),
            (0, 4),
            (1, 2),
            (2, 0),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
        ]
    )
    assert set(board.valid_moves(1)) == set(board.valid_moves(-1))
    board.add_stone(1, 2, -1)
    assert (
        board.to_numpy()
        == np.array(
            [
                [0, -1, 1, 0, 0],
                [-1, 0, -1, 1, -1],
                [0, -1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        )
    ).all()


def test_score_0() -> None:
    board: Board = Board(n=5)
    board.load_from_numpy(
        np.array(
            [
                [0, -1, 1, 0, 0],
                [-1, 0, -1, 1, 0],
                [0, -1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        )
    )
    s = board.get_scores()
    assert board.get_scores() == (4, 6)
    board.add_stone(1, 1, 1)
    # print(board.__str__())
    assert board.get_scores() == (6, 4)


def test_score_1() -> None:
    board: Board = Board(n=5)
    board.load_from_numpy(
        np.array(
            [
                [0, -1, 1, 0, 0],
                [-1, 0, 0, 1, 0],
                [0, -1, 0, 1, 0],
                [0, -1, 0, 1, 0],
                [0, 0, -1, 0, 1],
            ]
        )
    )
    # print(board.__str__())
    assert board.get_scores() == (10, 10)


def test_score_2() -> None:
    board: Board = Board(n=5)
    board.load_from_numpy(
        np.array(
            [
                [0, -1, 1, -1, 0],
                [-1, 0, 1, 1, -1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 1, 0],
                [0, 1, -1, 0, 1],
            ]
        )
    )
    # print(board.__str__())
    assert board.get_scores() == (13, 7)
