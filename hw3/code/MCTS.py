import math
import os
import pickle

import numpy as np
from util import *


class MCTS:
    def __init__(self, game, nnet, num_sims, C):
        self.game = game
        self.num_sims = num_sims
        self.nnet = nnet
        self.C = C
        self.training = True
        # stores Q=U/N values for (state,action)
        self.Q_state_action = {}
        # stores times edge (state,action) was visited
        self.N_state_action = {}
        # stores times board state was visited
        self.N_state = {}
        # stores policy
        self.P_state = {}

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_action_prob(self, board, player):
        board = self.game.get_board(board, player)
        if self.training:
            for i in range(self.num_sims):
                self.search(board)

        s = self.game.get_string(board)
        counts = np.array(
            [
                self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
                for a in range(self.game.action_size())
            ]
        )
        sum_count = counts.sum()
        if sum_count:
            probs = counts / sum_count
        else:
            probs = np.ones(len(counts), dtype=float) / len(counts)
        return probs

    def search(self, board):
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
        ##############################

        # TODO pick an action with the highest upper confidence bound (UCB)
        ##############################
        # YOUR CODE GOES HERE
        next_board = None  # compute the next board after executing the best action here
        ##############################

        v = self.search(next_board)

        # TODO update Q_state_action, N_state_action, and N_state
        ##############################
        # YOUR CODE GOES HERE
        ##############################

        return -v

    def save_params(self, file_name="mcts_param.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_params(self, file_name="mcts_param.pkl"):
        if not os.path.exists(file_name):
            print(f"Parameter file {file_name} does not exist, load failed!")
            return False
        with open(file_name, "rb") as f:
            self.__dict__ = pickle.load(f)
            print(f"Loaded parameters from {file_name}")
        return True
