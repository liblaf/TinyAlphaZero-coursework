import os
import pickle

import numpy as np
from TicTacToeGame import Board, TicTacToeGame


class MCTS:
    game: TicTacToeGame
    num_sims: int
    C: float
    training: bool
    Q_state_action: dict[
        tuple[str, int], float
    ]  # stores Q=U/N values for (state,action)
    N_state_action: dict[
        tuple[str, int], int
    ]  # stores #times edge (state,action) was visited
    N_state: dict[str, int]  # stores #times board state was visited

    def __init__(self, game: TicTacToeGame, num_sims: int, C: float):
        self.game = game
        self.num_sims = num_sims
        self.C = C
        self.training = True
        # stores Q=U/N values for (state,action)
        self.Q_state_action = {}
        # stores #times edge (state,action) was visited
        self.N_state_action = {}
        # stores #times board state was visited
        self.N_state = {}

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def get_action_prob(self, board: Board, player: int) -> np.ndarray:
        board = self.game.get_board(board, player)
        if self.training:
            for i in range(self.num_sims):
                self.search(board)

        s = self.game.get_string(board)
        counts = np.array(
            [
                self.N_state_action[(s, a)] if (s, a) in self.N_state_action else 0
                for a in range(self.game.action_size())
            ]
        )
        sum_count = counts.sum()
        if sum_count:
            probs = counts / sum_count
        else:
            probs = np.ones(len(counts), dtype=float) / len(counts)
        return probs

    def search(self, board: Board) -> float:
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound.
        """
        # use the current board state to get a unique string representation as a key
        s = self.game.get_string(board)

        # TODO handles leaf node
        ##############################
        # YOUR CODE GOES HERE
        is_terminal: float = self.game.is_terminal(board=board, player=1)
        if is_terminal != 0:
            return -is_terminal
        ##############################

        # TODO pick an action with the highest upper confidence bound (UCB)
        ##############################
        # YOUR CODE GOES HERE
        valid_moves: np.ndarray = self.game.get_valid_moves(board=board, player=1)
        upper_confidence_bounds: np.ndarray = np.zeros(shape=(self.game.action_size(),))
        for a in range(self.game.action_size()):
            if valid_moves[a]:
                if (s, a) in self.N_state_action:
                    upper_confidence_bounds[a] = self.Q_state_action[
                        (s, a)
                    ] + self.C * np.sqrt(
                        np.log(self.N_state[s]) / self.N_state_action[(s, a)]
                    )
                else:
                    action: int = a
                    break
            else:
                upper_confidence_bounds[a] = -np.inf
        else:
            action: int = np.argmax(upper_confidence_bounds)
        # compute the next board after executing the best action here
        next_board: Board = self.game.next_state(board=board, player=1, action=action)
        # flip board
        next_board: Board = self.game.get_board(board=next_board, player=-1)
        ##############################

        v = self.search(next_board)

        # TODO update Q_state_action, N_state_action, and N_state
        ##############################
        # YOUR CODE GOES HERE
        self.N_state[s] = self.N_state.get(s, 0) + 1
        self.Q_state_action[(s, action)] = (
            self.Q_state_action.get((s, action), 0)
            * self.N_state_action.get((s, action), 0)
            + v
        ) / (self.N_state_action.get((s, action), 0) + 1)
        self.N_state_action[(s, action)] = self.N_state_action.get((s, action), 0) + 1
        ##############################

        return -v

    def save_params(self, file_name: str = "mcts_param.pkl") -> None:
        with open(file_name, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_params(self, file_name: str = "mcts_param.pkl") -> bool:
        if not os.path.exists(file_name):
            print(f"Parameter file {file_name} does not exist, load failed!")
            return False
        with open(file_name, "rb") as f:
            self.__dict__ = pickle.load(f)
            print(f"Loaded parameters from {file_name}")
        return True
