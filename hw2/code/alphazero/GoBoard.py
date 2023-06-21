from typing import Any, List, Tuple

import numpy as np
from typing_extensions import Self

ADJACENCIES: List[Tuple[int, int]] = [(0, -1), (0, +1), (-1, 0), (+1, 0)]


class Stone:
    EMPTY: int = 0
    BLACK: int = 1
    WHITE: int = -1


class Board:
    last_captured: Tuple[int, int] = (-1, -1)
    last_move_is_pass: Tuple[bool, bool] = (False, False)
    num_moves: int = 0

    def __init__(self, n: int) -> None:
        assert n % 2 == 1
        self.n = n
        self.data = np.zeros(shape=(n, n))

    def __str__(self) -> str:
        ret = f"Num Moves: {self.num_moves}\n"
        ret += f"Last Captured: {self.last_captured[0]}, {self.last_captured[1]}\n"
        for i in range(self.n):
            for j in range(self.n):
                if self[i, j] == Stone.EMPTY:
                    ret += "+"
                elif self[i, j] == Stone.BLACK:
                    ret += "○"
                elif self[i, j] == Stone.WHITE:
                    ret += "●"
                else:
                    assert False, "Unreachable"  # pragma: no cover
            ret += "\n"
        return ret

    def __getitem__(self, *args, **kwargs) -> Any:
        return self.data.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs) -> None:
        return self.data.__setitem__(*args, **kwargs)

    def load_from_numpy(self, a: np.ndarray) -> None:
        assert a.shape == (self.n, self.n)
        self.data = a

    def to_numpy(self) -> np.ndarray:
        return self.data.copy()
        # Note: Copy if you don't want to mess up the original board.

    def add_stone(self, x: int, y: int, color: int) -> None:
        """
        Add a stone to the board, and remove captured stones
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        assert self._validate_move(x, y, color=color)
        self._unsafe_add_stone(x, y, color=color)

    def valid_moves(self, color: int) -> List[Tuple[int, int]]:
        """
        Return a list of avaliable moves
        @return: a list like [(0,0), (0,1), ...]
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        valid_moves: List[Tuple[int, int]] = []
        for x, y in np.ndindex(self.n, self.n):
            if not self._validate_move(x, y, color=color):
                continue
            valid_moves.append((x, y))
        return valid_moves

    def get_scores(self) -> Tuple[int, int]:
        """
        Compute score of players
        @return: a tuple (black_score, white_score)
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        black_score: int = 0
        white_score: int = 0
        for x, y in np.ndindex(self.n, self.n):
            if self[x, y] == Stone.EMPTY:
                color: int = Stone.EMPTY
                for i, j in self._get_connected(x, y, color=Stone.EMPTY):
                    for ii, jj in self._get_adjacencies(i, j):
                        if self[ii, jj] == Stone.EMPTY:
                            continue
                        if color == Stone.EMPTY:
                            color = self[ii, jj]
                        elif color == self[ii, jj]:
                            pass
                        elif color != self[ii, jj]:
                            color = Stone.EMPTY
                            break
                        else:
                            assert False, "Unreachable"  # pragma: no cover
                    else:
                        continue
                    break
                if color == Stone.EMPTY:
                    pass
                elif color == Stone.BLACK:
                    black_score += 1
                elif color == Stone.WHITE:
                    white_score += 1
                else:
                    assert False, "Unreachable"  # pragma: no cover
            elif self[x, y] == Stone.BLACK:
                black_score += 1
            elif self[x, y] == Stone.WHITE:
                white_score += 1
            else:
                assert False, "Unreachable"  # pragma: no cover
        return black_score, white_score

    def copy(self) -> Self:
        ret: Board = Board(self.n)
        ret.load_from_numpy(self.to_numpy())
        ret.last_captured = self.last_captured
        ret.last_move_is_pass = self.last_move_is_pass
        ret.num_moves = self.num_moves
        return ret

    def _within_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.n and 0 <= y < self.n

    def _get_adjacencies(self, x: int, y: int) -> List[Tuple[int, int]]:
        ret: List[Tuple[int, int]] = []
        for i, j in ADJACENCIES:
            if self._within_board(x + i, y + j):
                ret.append((x + i, y + j))
        return ret

    def _get_connected(self, x: int, y: int, color: int) -> List[Tuple[int, int]]:
        assert self[x, y] == color
        connected: np.ndarray = np.zeros(shape=(self.n, self.n), dtype=bool)
        connected[x, y] = True
        stack: List[Tuple[int, int]] = [(x, y)]
        while len(stack) > 0:
            x, y = stack.pop()
            adjacencies: List[Tuple[int, int]] = self._get_adjacencies(x, y)
            for i, j in adjacencies:
                if (self[i, j] == color) and (not connected[i, j]):
                    connected[i, j] = True
                    stack.append((i, j))
        ret: List[Tuple[int, int]] = []
        for x, y in np.ndindex(self.n, self.n):
            if connected[x, y]:
                ret.append((x, y))
        return ret

    def _get_liberties(self) -> np.ndarray:
        ret: np.ndarray = np.zeros(shape=(self.n, self.n), dtype=bool)
        for x, y in np.ndindex(self.n, self.n):
            if ret[x, y]:
                continue
            if self[x, y] == Stone.EMPTY:
                ret[x, y] = True
                for i, j in self._get_adjacencies(x, y):
                    if self[i, j] == Stone.EMPTY:
                        continue
                    for ii, jj in self._get_connected(i, j, color=self[i, j]):
                        ret[ii, jj] = True
        return ret

    def _unsafe_add_stone(self, x: int, y: int, color: int) -> None:
        self.num_moves += 1
        if (x, y) == (self.n, 0):
            self.last_move_is_pass = (True, self.last_move_is_pass[0])
            return
        self.last_move_is_pass = (False, self.last_move_is_pass[0])
        self[x, y] = color
        liberties: np.ndarray = self._get_liberties()
        captured: np.ndarray = (self.data == -color) & (liberties == False)
        if captured.sum() == 1:
            where: Tuple[np.ndarray, np.ndarray] = np.where(captured)
            self.last_captured = (where[0][0], where[1][0])
        else:
            self.last_captured = (-1, -1)
        self[captured] = Stone.EMPTY

    def _validate_move(self, x: int, y: int, color: int) -> bool:
        assert color in [Stone.BLACK, Stone.WHITE]
        if (x, y) == (self.n, 0):
            return True
        if self[x, y] != Stone.EMPTY:
            return False
        board: Board = self.copy()
        board._unsafe_add_stone(x, y, color=color)
        if not board._get_liberties().all():
            return False
        if (x, y) == self.last_captured:
            return False
        return True
