from typing import Tuple

import numpy as np

from .GoBoard import Board


class GoGame:
    def __init__(self, n: int = 3) -> None:
        assert n % 2 == 1
        self.n = n

    def reset(self) -> Board:
        """
        Reset the game.
        """
        return Board(self.n)

    def obs_size(self) -> Tuple[int, int]:
        """
        Size of the board.
        """
        return self.n, self.n

    def action_size(self) -> int:
        """
        Number of all possible actions.
        """
        return (
            self.n * self.n + 1
        )  # the extra 1 is for 'pass'(虚着), a action for doing nothing

    def get_board(self, board: Board, player: int) -> Board:
        """
        Convert given board to a board from player1's perspective.
        If current player is player1, do nothing, else, reverse the board.
        This can help you write neater code for search algorithms as you can go for the maximum return every step.

        @param board: the board to convert
        @param player: 1 or -1, the player of current board

        @return: a board from player1's perspective
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        new_board: Board = board.copy()
        new_board.load_from_numpy(board.to_numpy() * player)
        return new_board

    def next_state(self, board: Board, player: int, action: int) -> Board:
        """
        Get the next state by executing the action for player.
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        # Note: don't forget 'pass'
        x, y = action // self.n, action % self.n
        new_board: Board = board.copy()
        new_board.add_stone(x, y, color=player)
        return new_board

    def get_valid_moves(self, board: Board, player: int) -> np.ndarray:
        """
        Get a binary vector of length self.action_size(), 1 for all valid moves.
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        valid_moves = np.zeros(shape=(self.action_size(),), dtype=bool)
        for i, j in board.valid_moves(color=player):
            valid_moves[i * self.n + j] = True
        valid_moves[-1] = True
        return valid_moves

    @staticmethod
    def is_terminal(board: Board, player: int) -> float:
        """
        Check whether the game is over.
        @return: 1 or -1 if player or opponent wins, 1e-4 if draw, 0 if not over
        """

        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        # Note: end game when two consecutive passes or too many moves, compare scores of two players.
        def get_reward(board: Board, player: int) -> float:
            assert player in [+1, -1]
            black_score, white_score = board.get_scores()
            if black_score > white_score:
                return player
            elif black_score < white_score:
                return -player
            elif black_score == white_score:
                return 1e-4
            else:
                assert False, "Unreachable"  # pragma: no cover

        if board.last_move_is_pass == (True, True):
            return get_reward(board=board, player=player)
        if board.num_moves > board.n * board.n * 4:
            return get_reward(board=board, player=player)
        return 0

    @staticmethod
    def get_string(board: Board) -> str:
        """
        Convert the board to a string.
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        # Note: different board(Game Statue) must return different string
        return str(board)

    @staticmethod
    def display(board: Board) -> None:
        """
        Print the board to console.
        """
        print(str(board))
