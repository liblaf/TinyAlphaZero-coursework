import numpy as np


class Board:
    w: int
    h: int
    n: int

    def __init__(self, h: int = 3, w: int = 3, n: int = 3) -> None:
        self.w = w
        self.h = h
        self.n = n
        self.pieces = np.zeros((self.h, self.w), dtype=int)

    def get_legal_moves(self, player: int) -> list[tuple[int, int]]:
        """
        Return the legal moves for the given player.
        @param player: 1 or -1
        """
        moves = []
        for x in range(self.h):
            for y in range(self.w):
                if self.pieces[x][y] == 0:
                    moves.append((x, y))
        return moves

    def has_legal_moves(self) -> bool:
        return (self.pieces == 0).any()

    def check_win_status(self, player: int) -> bool:
        """
        Check whether given player has won.
        @param player: 1 or -1, indicating which player to check
        """
        for h in range(self.h):
            for w in range(self.w):
                if self.pieces[h][w] != player:
                    continue
                if (
                    h in range(self.h - self.n + 1)
                    and len(set(self.pieces[i][w] for i in range(h, h + self.n))) == 1
                ):
                    return True
                if (
                    w in range(self.w - self.n + 1)
                    and len(set(self.pieces[h][j] for j in range(w, w + self.n))) == 1
                ):
                    return True
                if (
                    h in range(self.h - self.n + 1)
                    and w in range(self.w - self.n + 1)
                    and len(set(self.pieces[h + k][w + k] for k in range(self.n))) == 1
                ):
                    return True
                if (
                    h in range(self.h - self.n + 1)
                    and w in range(self.n - 1, self.w)
                    and len(set(self.pieces[h + l][w - l] for l in range(self.n))) == 1
                ):
                    return True

        return False

    def execute_move(self, move: tuple[int, int], player: int) -> None:
        x, y = move
        assert self.pieces[x][y] == 0
        self.pieces[x][y] = player


class TicTacToeGame:
    def __init__(self, h=3, w=3, n=3) -> None:
        self.w = w
        self.h = h
        self.n = n

    def reset(self) -> Board:
        """
        Reset the game.
        """
        return Board(self.h, self.w, self.n)

    def get_board(self, board: Board, player: int) -> Board:
        """
        Convert given board to a board from player1's perspective.
        If current player is player1, do nothing, else, reverse the board.
        This can help you write neater code for search algorithms as you can go for the maximum return every step.

        @param board: the board to convert
        @param player: 1 or -1, the player of current board

        @return: a board from player1's perspective
        """
        b = Board(self.h, self.w, self.n)
        b.pieces = board.pieces * player
        return b

    def obs_size(self) -> tuple[int, int]:
        """
        Size of the board.
        """
        return self.h, self.w

    def action_size(self) -> int:
        """
        Number of all possible actions.
        """
        return self.h * self.w

    def next_state(self, board: Board, player: int, action: int) -> Board:
        """
        Get the next state by executing the action for player.
        """
        b = Board(self.h, self.w, self.n)
        b.pieces = board.pieces.copy()
        move = (action // self.w, action % self.w)
        b.execute_move(move, player)
        return b

    def get_valid_moves(self, board: Board, player: int) -> np.ndarray:
        """
        Get a binary vector of length self.action_size(), 1 for all valid moves.
        """
        is_valid_move = [0] * self.action_size()
        legal_moves = board.get_legal_moves(player)
        for x, y in legal_moves:
            is_valid_move[self.w * x + y] = 1
        return np.array(is_valid_move)

    @staticmethod
    def is_terminal(board: Board, player: int) -> float:
        """
        Check whether the game is over.
        @return: 1 or -1 if player or opponent wins, 1e-4 if draw, 0 if not over
        """
        if board.check_win_status(player):
            return 1
        if board.check_win_status(-player):
            return -1
        if board.has_legal_moves():
            return 0
        return 1e-4

    @staticmethod
    def get_string(board: Board) -> str:
        """
        Convert the board to a string.
        """
        return np.array2string(board.pieces)

    @staticmethod
    def display(board: Board) -> None:
        """
        Print the board to console.
        """
        width = board.w
        height = board.h

        print()
        for x in range(width):
            print("{0:8}".format(x), end="")
        print("\r\n")
        for i in range(height):
            print("{0:4d}".format(i), end="")
            for j in range(width):
                p = board.pieces[i, j]
                if p == -1:
                    print("X".center(8), end="")
                elif p == 1:
                    print("O".center(8), end="")
                else:
                    print("_".center(8), end="")
            print("\r\n\r\n")
