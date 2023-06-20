from enum import IntEnum
from typing import Any, Iterable

import numpy as np
from typing_extensions import Self


class Stone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = -1


class Board:
    data: np.ndarray
    last_capture: tuple[int, int] = (-1, -1)
    last_move_is_pass: tuple[bool, bool] = (False, False)
    n: int
    num_moves: int = 0

    def __init__(self, n: int) -> None:
        assert n % 2 == 1
        self.n = n
        self.data = np.zeros(shape=(n, n))

    def __str__(self) -> str:
        ret: str = ""
        ret += f"Last Capture: {self.last_capture}" + "\n"
        ret += f"Num  Moves  : {self.num_moves}" + "\n"
        for i in range(self.n):
            for j in range(self.n):
                match self.data[i, j]:
                    case Stone.EMPTY:
                        ret += "+"
                    case Stone.BLACK:
                        ret += "\N{Black Circle}"  # ●
                    case Stone.WHITE:
                        ret += "\N{White Circle}"  # ○
                    case _:
                        assert False, "Unreachable"  # pragma: no cover
            ret += "\n"
        return ret

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = value

    def load_from_numpy(self, a: np.ndarray) -> None:
        assert a.shape == (self.n, self.n)
        self.data = a

    def to_numpy(self) -> np.ndarray:
        return self.data.copy()
        # Note: Copy if you don't want to mess up the original board.

    def add_stone(self, x: int, y: int, color: Stone | int) -> None:
        """
        Add a stone to the board, and remove captured stones
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        assert self._validate_move(x, y, color=Stone(color))
        self._unsafe_add_stone(x, y, color=Stone(color))

    def valid_moves(self, color: Stone | int) -> list[tuple[int, int]]:
        """
        Return a list of avaliable moves
        @return: a list like [(0,0), (0,1), ...]
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        return list(self._get_valid_moves(color=Stone(color)))

    def get_scores(self) -> tuple[int, int]:
        """
        Compute score of players
        @return: a tuple (black_score, white_score)
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        black_score: int = 0
        white_score: int = 0
        visited: np.ndarray = np.zeros(shape=(self.n, self.n), dtype=bool)
        for x, y in np.ndindex(self.n, self.n):
            match self.data[x, y]:
                case Stone.EMPTY:
                    if visited[x, y]:
                        continue
                    color: Stone = Stone.EMPTY
                    connected: list[tuple[int, int]] = list(self._get_connected(x, y))
                    for nx, ny in self._get_adjacencies(connected):
                        if self.data[nx, ny] == Stone.EMPTY:
                            pass
                        else:
                            if color == Stone.EMPTY:
                                color = self.data[nx, ny]
                            elif color == self.data[nx, ny]:
                                pass
                            elif color != self.data[nx, ny]:
                                color = Stone.EMPTY
                                break
                            else:
                                assert False, "Unreachable"  # pragma: no cover
                    match color:
                        case Stone.EMPTY:
                            pass
                        case Stone.BLACK:
                            black_score += len(connected)
                        case Stone.WHITE:
                            white_score += len(connected)
                    for x, y in connected:
                        visited[x, y] = True
                case Stone.BLACK:
                    black_score += 1
                case Stone.WHITE:
                    white_score += 1
        return black_score, white_score

    def copy(self) -> Self:
        board: Board = Board(n=self.n)
        board.data = self.data.copy()
        board.last_capture = self.last_capture
        board.last_move_is_pass = self.last_move_is_pass
        board.num_moves = self.num_moves
        return board

    def _within_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.n and 0 <= y < self.n

    def _get_adjacency(self, x: int, y: int) -> Iterable[tuple[int, int]]:
        for dx, dy in [(-1, 0), (0, -1), (0, +1), (+1, 0)]:
            nx, ny = x + dx, y + dy
            if self._within_board(nx, ny):
                yield nx, ny

    def _get_connected(self, x: int, y: int) -> Iterable[tuple[int, int]]:
        color: Stone = self[x, y]
        stack: list[tuple[int, int]] = [(x, y)]
        visited: np.ndarray = np.zeros(shape=(self.n, self.n), dtype=bool)
        while len(stack) > 0:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            yield x, y
            visited[x, y] = True
            for nx, ny in self._get_adjacency(x, y):
                if self[nx, ny] == color:
                    stack.append((nx, ny))

    def _get_adjacencies(
        self, connected: Iterable[tuple[int, int]]
    ) -> Iterable[tuple[int, int]]:
        for x, y in connected:
            for nx, ny in self._get_adjacency(x, y):
                yield nx, ny

    def _get_liberties(self) -> np.ndarray:
        ret: np.ndarray = np.zeros(shape=(self.n, self.n), dtype=bool)
        visited: np.ndarray = np.zeros(shape=(self.n, self.n), dtype=bool)
        for x, y in np.ndindex(self.n, self.n):
            match self.data[x, y]:
                case Stone.EMPTY:
                    ret[x, y] = True
                case color:
                    if visited[x, y]:
                        continue
                    connected: list[tuple[int, int]] = list(self._get_connected(x, y))
                    for nnx, nny in self._get_adjacencies(connected):
                        match self.data[nnx, nny]:
                            case Stone.EMPTY:
                                for nx, ny in connected:
                                    ret[nx, ny] = True
                                break
                            case _:
                                pass
                    for nx, ny in connected:
                        visited[nx, ny] = True
        return ret

    @property
    def liberties(self) -> np.ndarray:
        return self._get_liberties()

    def _unsafe_add_stone(self, x: int, y: int, color: Stone) -> None:
        self.num_moves += 1
        if (x, y) == (self.n, 0):
            self.last_move_is_pass = (self.last_move_is_pass[-1], True)
        else:
            self.last_move_is_pass = (self.last_move_is_pass[-1], False)
            self.data[x, y] = color
            capture: np.ndarray = (self.liberties == False) & (self.data == -color)
            if capture.sum() == 1:
                where: tuple[np.ndarray, np.ndarray] = np.where(capture)
                self.last_capture = where[0][0], where[1][0]
            else:
                self.last_capture = (-1, -1)
            self.data[capture] = Stone.EMPTY

    def _validate_move(self, x: int, y: int, color: Stone) -> bool:
        assert color in [Stone.BLACK, Stone.WHITE]
        if (x, y) == (self.n, 0):
            return True
        assert self._within_board(x, y)
        if self.data[x, y] != Stone.EMPTY:
            return False
        if (x, y) == self.last_capture:
            return False
        board: Board = self.copy()
        board._unsafe_add_stone(x, y, color=color)
        if not board.liberties.all():
            return False
        return True

    def _get_valid_moves(self, color: Stone) -> Iterable[tuple[int, int]]:
        for x, y in np.ndindex(self.n, self.n):
            if self._validate_move(x, y, color=color):
                yield (x, y)
