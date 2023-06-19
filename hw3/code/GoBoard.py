import numpy as np


class Stone:
    EMPTY = 0
    BLACK = 1
    WHITE = -1


class Board:
    def __init__(self, n):
        assert n % 2 == 1
        self.n = n
        self.data = np.zeros((n, n))

    def __str__(self) -> str:
        ret = ""
        for i in range(self.n):
            for j in range(self.n):
                if self[i, j] == Stone.EMPTY:
                    ret += "+"
                elif self[i, j] == Stone.BLACK:
                    ret += "○"
                else:
                    ret += "●"
            ret += "\n"
        return ret

    def load_from_numpy(self, a: np.ndarray):
        assert a.shape == (self.n, self.n)
        self.data = a

    def to_numpy(self):
        return self.data.copy()
        # Note: Copy if you don't want to mess up the original board.

    def add_stone(self, x, y, color):
        """
        Add a stone to the board, and remove captured stones
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        pass

    def valid_moves(self, color):
        """
        Return a list of avaliable moves
        @return: a list like [(0,0), (0,1), ...]
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        return []

    def get_scores(self):
        """
        Compute score of players
        @return: a tuple (black_score, white_score)
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        return (0, 0)
