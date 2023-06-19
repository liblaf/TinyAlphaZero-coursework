import datetime
import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from MCTS import MCTS
from pit import test_multi_match
from Player import FastEvalPlayer
from tqdm import tqdm

log = logging.getLogger(__name__)


class Trainer:
    """ """

    def __init__(self, game, nnet, config):
        self.game = game
        self.next_net = nnet
        self.last_net = self.next_net.__class__(self.game)
        self.config = config
        self.mcts = MCTS(
            self.game, self.next_net, self.config.num_sims, self.config.cpuct
        )
        self.train_data_packs = []

    def collect_single_game(self):
        """
        Collect self-play data for one game.

        @return game_history: A list of (board, pi, z)
        """
        # create a New MCTS
        self.mcts = MCTS(
            self.game, self.next_net, self.config.num_sims, self.config.cpuct
        )
        self.mcts.train()

        game_history = []
        board = self.game.reset()
        current_player = 1
        current_step = 0

        # self-play until the game is ended
        while True:
            current_step += 1
            pi = self.mcts.get_action_prob(board, current_player)
            datas = self.game.get_transform_data(
                self.game.get_board(board, current_player), pi
            )
            for b, p in datas:
                game_history.append([b, current_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board = self.game.next_state(board, current_player, action)
            current_player *= -1
            game_result = self.game.is_terminal(board, current_player)

            if game_result != 0:  # Game Ended
                return [
                    (x[0], x[2], game_result * ((-1) ** (x[1] != current_player)))
                    for x in game_history
                ]

    def train(self):
        """
        Main Training Loop of AlphaZero
        each iteration:
            * Collect data by self play
            * Train the network
            * Pit the new model against the old model
                If the new model wins, save the new model, and evaluate the new model
                Otherwise, reject the new model and keep the old model

        """
        for i in range(1, self.config.max_training_iter + 1):
            log.info(f"Starting Iter #{i} ...")

            data_pack = deque([], maxlen=self.config.max_train_data_packs_len + 1)
            T = tqdm(range(self.config.selfplay_each_iter), desc="Self Play")
            for _ in T:
                game_data = self.collect_single_game()
                data_pack += game_data
                r = game_data[0][-1]
                T.set_description_str(f"Self Play win={r}, len={len(game_data)}")

            self.train_data_packs.append(data_pack)

            if len(self.train_data_packs) > self.config.max_train_data_packs_len:
                log.warning(f"Removing the oldest data pack...")
                self.train_data_packs.pop(0)

            trainExamples = []
            for e in self.train_data_packs:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.next_net.save_checkpoint(
                folder=self.config.checkpoint_folder, filename="temp.pth.tar"
            )
            self.last_net.load_checkpoint(
                folder=self.config.checkpoint_folder, filename="temp.pth.tar"
            )

            self.next_net.train(trainExamples)

            next_mcts = MCTS(
                self.game, self.next_net, self.config.num_sims, self.config.cpuct
            )
            last_mcts = MCTS(
                self.game, self.last_net, self.config.num_sims, self.config.cpuct
            )

            log.info("Pitting against last version...")
            ######################################
            #        YOUR CODE GOES HERE         #
            ######################################
            # Pitting against last version, and decide whether to save the new model
