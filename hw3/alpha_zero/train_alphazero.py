import logging
from collections import deque
from pathlib import Path
from random import shuffle

import numpy as np
from tqdm import tqdm

from .GoBoard import Board
from .GoGame import GoGame
from .GoNNet import GoNNetWrapper
from .MCTS import MCTS
from .pit import test_multi_match
from .Player import FastEvalPlayer
from .utils import DotDict

log = logging.getLogger(__name__)


class Trainer:
    """ """

    game: GoGame
    next_net: GoNNetWrapper
    last_net: GoNNetWrapper
    config: DotDict
    mcts: MCTS

    def __init__(self, game: GoGame, nnet: GoNNetWrapper, config: DotDict) -> None:
        num_sims: int = config["num_sims"]
        cpuct: float = config["cpuct"]
        self.game = game
        self.next_net = nnet
        self.last_net = self.next_net.__class__(self.game)
        self.config = config
        self.mcts = MCTS(game=self.game, nnet=self.next_net, num_sims=num_sims, C=cpuct)
        self.train_data_packs = []

    def collect_single_game(self) -> list[tuple[Board, np.ndarray, float]]:
        """
        Collect self-play data for one game.

        @return game_history: A list of (board, pi, z)
        """
        num_sims: int = self.config["num_sims"]
        cpuct: float = self.config["cpuct"]
        # create a New MCTS
        self.mcts = MCTS(game=self.game, nnet=self.next_net, num_sims=num_sims, C=cpuct)
        self.mcts.train()

        game_history: list[list] = []
        board: Board = self.game.reset()
        current_player: int = 1
        current_step: int = 0

        # self-play until the game is ended
        while True:
            current_step += 1
            pi: np.ndarray = self.mcts.get_action_prob(
                board=board, player=current_player
            )
            datas: list[tuple[Board, np.ndarray]] = self.game.get_transform_data(
                self.game.get_board(board, current_player), pi
            )
            for b, p in datas:
                game_history.append([b, current_player, p, None])

            action: int = np.random.choice(len(pi), p=pi)
            board: Board = self.game.next_state(
                board=board, player=current_player, action=action
            )
            current_player *= -1
            game_result: float = self.game.is_terminal(
                board=board, player=current_player
            )

            if game_result != 0:  # Game Ended
                return [
                    (x[0], x[2], game_result * ((-1) ** (x[1] != current_player)))
                    for x in game_history
                ]

    def train(self) -> None:
        """
        Main Training Loop of AlphaZero
        each iteration:
            * Collect data by self play
            * Train the network
            * Pit the new model against the old model
                If the new model wins, save the new model, and evaluate the new model
                Otherwise, reject the new model and keep the old model

        """
        max_training_iter: int = self.config["max_training_iter"]
        max_train_data_packs_len: int = self.config["max_train_data_packs_len"]
        selfplay_each_iter: int = self.config["selfplay_each_iter"]
        checkpoint_folder: str | Path = self.config["checkpoint_folder"]
        num_sims: int = self.config["num_sims"]
        cpuct: float = self.config["cpuct"]
        update_threshold: float = self.config["update_threshold"]

        for i in range(1, max_training_iter + 1):
            log.info(f"Starting Iter #{i} ...")

            data_pack = deque([], maxlen=max_train_data_packs_len + 1)
            T = tqdm(range(selfplay_each_iter), desc="Self Play")
            for _ in T:
                game_data = self.collect_single_game()
                data_pack += game_data
                r = game_data[0][-1]
                T.set_description_str(f"Self Play win={r}, len={len(game_data)}")

            self.train_data_packs.append(data_pack)

            if len(self.train_data_packs) > max_train_data_packs_len:
                log.warning(f"Removing the oldest data pack...")
                self.train_data_packs.pop(0)

            trainExamples = []
            for e in self.train_data_packs:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.next_net.save_checkpoint(
                folder=checkpoint_folder, filename="temp.pth.tar"
            )
            self.last_net.load_checkpoint(
                folder=checkpoint_folder, filename="temp.pth.tar"
            )

            self.next_net.train(trainExamples)

            next_mcts = MCTS(self.game, self.next_net, num_sims, cpuct)
            last_mcts = MCTS(self.game, self.last_net, num_sims, cpuct)

            log.info("Pitting against last version...")
            ######################################
            #        YOUR CODE GOES HERE         #
            ######################################
            # Pitting against last version, and decide whether to save the new model
            next_player: FastEvalPlayer = FastEvalPlayer(next_mcts)
            last_player: FastEvalPlayer = FastEvalPlayer(last_mcts)
            next_win, last_win, draw = test_multi_match(
                player1=next_player, player2=last_player, game=self.game
            )
            if next_win / (next_win + last_win + draw) > update_threshold:
                log.info("Accept new model.")
                self.next_net.save_checkpoint(
                    folder=checkpoint_folder, filename=f"checkpoint_{i}.pth.tar"
                )
                self.next_net.save_checkpoint(
                    folder=checkpoint_folder, filename="best.pth.tar"
                )
            else:
                log.info("Reject new model.")
                self.next_net.load_checkpoint(
                    folder=checkpoint_folder, filename="temp.pth.tar"
                )
