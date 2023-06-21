from typing import List

from ..GoGame import GoGame
from ..pit_multiprocessing import multi_match, single_match
from ..Player import RandomPlayer


def test_single_match(board_size: int = 9) -> None:
    game: GoGame = GoGame(n=board_size)
    player_1: RandomPlayer = RandomPlayer(game=game, player=1)
    player_2: RandomPlayer = RandomPlayer(game=game, player=-1)
    scores: List[int] = single_match(player_1, player_2, game)
    assert len(scores) == 3
    assert all([score in [0, 1] for score in scores])
    assert sum(scores) == 1


def test_multi_match(board_size: int = 9, n_test: int = 100) -> None:
    game: GoGame = GoGame(n=board_size)
    player_1: RandomPlayer = RandomPlayer(game=game, player=1)
    player_2: RandomPlayer = RandomPlayer(game=game, player=-1)
    player_1_win, draw, player_2_win = multi_match(
        player_1, player_2, game, n_test=n_test
    )
    assert player_1_win + draw + player_2_win == n_test
