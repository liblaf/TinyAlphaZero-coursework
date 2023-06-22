import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from . import GoNNetWrapper
from .GoBoard import Board
from .GoGame import GoGame


class MCTS:
    game: GoGame
    nnet: GoNNetWrapper
    num_sims: int
    C: float
    training: bool = True
    visit_count_state: Dict[str, int] = {}
    visit_count_state_action: Dict[Tuple[str, int], int] = {}
    # total_action_value: Dict[Tuple[str, int], float] = {}
    mean_action_value: Dict[Tuple[str, int], float] = {}
    prior_probability: Dict[str, np.ndarray] = {}

    def __init__(
        self, game: GoGame, nnet: GoNNetWrapper, num_sims: int, C: float
    ) -> None:
        self.game = game
        self.num_sims = num_sims
        self.nnet = nnet
        self.C = C
        self.training = True
        # stores Q=U/N values for (state,action)
        # self.Q_state_action = {}
        # stores times edge (state,action) was visited
        # self.N_state_action = {}
        # stores times board state was visited
        # self.N_state = {}
        # stores policy
        # self.P_state = {}

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def get_action_prob(self, board: Board, player: int) -> np.ndarray:
        board = self.game.get_board(board, player)
        if self.training:
            for i in range(self.num_sims):
                self.search(board)

        s: str = self.game.get_string(board)
        counts: np.ndarray = np.array(
            [
                self.visit_count_state_action.get((s, a), 0)
                for a in range(self.game.action_size())
            ]
        )
        valid_moves: np.ndarray = self.game.get_valid_moves(board, player)
        counts *= valid_moves
        sum_count: int = counts.sum()
        if sum_count:
            probs: np.ndarray = counts / sum_count
        else:
            probs: np.ndarray = valid_moves / valid_moves.sum()
        return probs

    def search(self, board: Board) -> float:
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound.
        """
        # use the current board state to get a unique string representation as a key
        state: str = self.game.get_string(board)

        # TODO handles leaf node
        ##############################
        # YOUR CODE GOES HERE
        ##############################
        value: float = self.game.is_terminal(board=board, player=1)
        if value != 0:
            return -value
        if state not in self.prior_probability:
            probability, value_array = self.nnet.predict(board.data)
            value = value_array[0]
            probability *= self.game.get_valid_moves(board=board, player=1)
            sum_probability: float = probability.sum()
            if sum_probability > 0:
                probability = probability / sum_probability
            self.prior_probability[state] = probability
            return -value

        # TODO pick an action with the highest upper confidence bound (UCB)
        ##############################
        # YOUR CODE GOES HERE
        ##############################
        valid_moves: np.ndarray = self.game.get_valid_moves(board=board, player=1)
        best_action: int = -1
        current_best: float = -np.inf
        for action in range(self.game.action_size()):
            if not valid_moves[action]:
                continue
            upper_confidence_bound: float = self.mean_action_value.get(
                (state, action), 0
            ) + self.C * self.prior_probability[state][action] * np.sqrt(
                self.visit_count_state.get(state, 0)
                / (1 + self.visit_count_state_action.get((state, action), 0))
            )
            if upper_confidence_bound > current_best:
                best_action = action
                current_best = upper_confidence_bound
        action: int = best_action
        # compute the next board after executing the best action here
        next_board: Board = self.game.next_state(board=board, player=1, action=action)
        next_board: Board = self.game.get_board(next_board, player=-1)

        value = self.search(next_board)

        # TODO update Q_state_action, N_state_action, and N_state
        ##############################
        # YOUR CODE GOES HERE
        ##############################
        if (state, action) in self.mean_action_value:
            self.mean_action_value[(state, action)] = (
                self.visit_count_state_action[(state, action)]
                * self.mean_action_value[(state, action)]
                + value
            ) / (self.visit_count_state_action[(state, action)] + 1)
            self.visit_count_state_action[(state, action)] += 1
        else:
            self.visit_count_state_action[(state, action)] = 1
            self.mean_action_value[(state, action)] = value
        self.visit_count_state[state] = self.visit_count_state.get(state, 0) + 1

        return -value

    def save_params(self, file_name: Union[str, Path] = "mcts_param.pkl") -> None:
        with open(file_name, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_params(self, file_name: Union[str, Path] = "mcts_param.pkl") -> bool:
        if not os.path.exists(file_name):
            print(f"Parameter file {file_name} does not exist, load failed!")
            return False
        with open(file_name, "rb") as f:
            self.__dict__ = pickle.load(f)
            print(f"Loaded parameters from {file_name}")
        return True
