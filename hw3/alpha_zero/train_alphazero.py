import logging
import os
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Tuple, Union

import anylearn
import numpy as np
import torch.multiprocessing as mp

from . import PROCESSES, GoNNetWrapper
from .GoBoard import Board
from .GoGame import GoGame
from .MCTS import MCTS
from .pit import multi_match as multi_match_sequential
from .pit_multiprocessing import multi_match as multi_match_multiprocessing
from .Player import FastEvalPlayer, Player
from .plot import plot_loss, plot_model_update_frequency, plot_win_rate

log = logging.getLogger(__name__)


def static_collect_single_game(
    board_size: int,
    nnet: GoNNetWrapper,
    num_sims: int,
    cpuct: float,
) -> List[Tuple[Board, np.ndarray, float]]:
    """
    Collect self-play data for one game.

    @return game_history: A list of (board, pi, z)
    """
    # create a New MCTS
    game: GoGame = GoGame(n=board_size)
    mcts = MCTS(game=game, nnet=nnet, num_sims=num_sims, C=cpuct)
    mcts.train()

    game_history: List[list] = []
    board: Board = game.reset()
    current_player: int = 1
    current_step: int = 0

    # self-play until the game is ended
    while True:
        current_step += 1
        pi: np.ndarray = mcts.get_action_prob(board=board, player=current_player)
        datas: List[Tuple[Board, np.ndarray]] = game.get_transform_data(
            game.get_board(board, current_player), pi
        )
        for b, p in datas:
            game_history.append([b, current_player, p, None])

        action: int = np.random.choice(len(pi), p=pi)
        board: Board = game.next_state(
            board=board, player=current_player, action=action
        )
        current_player *= -1
        game_result: float = game.is_terminal(board=board, player=current_player)

        if game_result != 0:  # Game Ended
            return [
                (x[0], x[2], game_result * ((-1) ** (x[1] != current_player)))
                for x in game_history
            ]


class Trainer:
    """ """

    game: GoGame
    next_net: GoNNetWrapper
    last_net: GoNNetWrapper
    config: Dict[str, Any]
    mcts: MCTS
    train_data_packs: List[Iterable[Tuple[Board, np.ndarray, float]]]

    def __init__(
        self, game: GoGame, nnet: GoNNetWrapper, config: Dict[str, Any]
    ) -> None:
        num_sims: int = config["num_sims"]
        cpuct: float = config["cpuct"]
        self.game = game
        self.next_net = nnet
        self.last_net = self.next_net.__class__(self.game)
        self.config = config
        self.mcts = MCTS(game=self.game, nnet=self.next_net, num_sims=num_sims, C=cpuct)
        self.train_data_packs = []

    def collect_single_game(self) -> List[Tuple[Board, np.ndarray, float]]:
        """
        Collect self-play data for one game.

        @return game_history: A list of (board, pi, z)
        """
        num_sims: int = self.config["num_sims"]
        cpuct: float = self.config["cpuct"]
        # create a New MCTS
        self.mcts = MCTS(game=self.game, nnet=self.next_net, num_sims=num_sims, C=cpuct)
        self.mcts.train()

        game_history: List[list] = []
        board: Board = self.game.reset()
        current_player: int = 1
        current_step: int = 0

        # self-play until the game is ended
        while True:
            current_step += 1
            pi: np.ndarray = self.mcts.get_action_prob(
                board=board, player=current_player
            )
            datas: List[Tuple[Board, np.ndarray]] = self.game.get_transform_data(
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
        checkpoint_folder: Union[str, Path] = self.config["checkpoint_folder"]
        num_sims: int = self.config["num_sims"]
        cpuct: float = self.config["cpuct"]
        update_threshold: float = self.config["update_threshold"]
        pit_with: Player = self.config["pit_with"]
        multiprocessing: bool = self.config["multiprocessing"]
        loss_history: List[Tuple[float, float]] = []
        self_pit_results: List[Tuple[float, int, int, int]] = []
        pit_results: List[Tuple[float, int, int, int]] = []

        os.makedirs(name=checkpoint_folder, exist_ok=True)
        start_time: datetime = datetime.now()
        (Path(checkpoint_folder) / "start-time.txt").write_text(start_time.isoformat())

        for i in range(1, max_training_iter + 1):
            log.info(f"Starting Iter #{i} ...")

            data_pack: Deque[Tuple[Board, np.ndarray, float]] = deque()

            if multiprocessing:
                self.next_net.nnet.share_memory()
                try:
                    mp.set_start_method("spawn")
                except RuntimeError:
                    pass
                with mp.Pool(processes=PROCESSES) as pool:
                    for game_data in pool.starmap(
                        func=static_collect_single_game,
                        iterable=[
                            (
                                self.game.n,
                                self.next_net,
                                num_sims,
                                cpuct,
                            )
                        ]
                        * selfplay_each_iter,
                    ):
                        data_pack += game_data
                        r = game_data[0][-1]
                        # log.info(f"Self Play win={r}, len={len(game_data)}")
            else:
                for i in range(1, selfplay_each_iter + 1):
                    game_data = self.collect_single_game()
                    data_pack += game_data
                    r = game_data[0][-1]
                    # log.info(
                    #     f"Self Play {i}/{selfplay_each_iter}: win={r}, len={len(game_data)}"
                    # )

            self.train_data_packs.append(data_pack)

            if len(self.train_data_packs) > max_train_data_packs_len:
                log.warning(f"Removing the oldest data pack...")
                self.train_data_packs.pop(0)

            train_examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for e in self.train_data_packs:
                train_examples.extend(
                    (data_pack[0].data, data_pack[1], np.array([data_pack[2]]))
                    for data_pack in e
                )
            random.shuffle(train_examples)

            self.next_net.save_checkpoint(
                folder=checkpoint_folder, filename="temp.pth.tar"
            )
            self.last_net.load_checkpoint(
                folder=checkpoint_folder, filename="temp.pth.tar"
            )

            loss_history += self.next_net.train(train_examples)
            log.info("Loss: %s", loss_history[-1][1])
            np.savetxt(
                fname=Path(checkpoint_folder) / "loss-history.csv",
                X=loss_history,
                delimiter=",",
            )
            plot_loss(
                start_time=start_time,
                loss_history=loss_history,
                output=Path(checkpoint_folder) / "loss.png",
            )

            next_mcts = MCTS(self.game, self.next_net, num_sims, cpuct)
            last_mcts = MCTS(self.game, self.last_net, num_sims, cpuct)

            log.info("Pitting against last version...")
            ######################################
            #        YOUR CODE GOES HERE         #
            ######################################
            # Pitting against last version, and decide whether to save the new model
            next_player: FastEvalPlayer = FastEvalPlayer(next_mcts)
            last_player: FastEvalPlayer = FastEvalPlayer(last_mcts)
            multi_match = (
                multi_match_multiprocessing
                if multiprocessing
                else multi_match_sequential
            )
            next_win, last_win, draw = multi_match(
                player1=next_player, player2=last_player, game=self.game
            )
            self_pit_results.append(
                (datetime.now().timestamp(), next_win, last_win, draw)
            )
            plot_model_update_frequency(
                start_time=start_time,
                self_pit_results=self_pit_results,
                output=Path(checkpoint_folder) / "model-update-frequency.png",
            )

            def save_pit(win: int, lose: int, draw: int) -> None:
                pit_results.append((datetime.now().timestamp(), win, lose, draw))
                np.savetxt(
                    fname=Path(checkpoint_folder) / "pit-results.csv",
                    X=pit_results,
                    delimiter=",",
                )
                plot_win_rate(
                    start_time=start_time,
                    pit_results=pit_results,
                    output=Path(checkpoint_folder) / "win-rate.png",
                )

            def pit() -> None:
                win, lose, draw = multi_match(
                    player1=next_player, player2=pit_with, game=self.game
                )
                log.info(f"PIT with {pit_with} Win: {win}, Lose: {lose}, Draw: {draw}")
                save_pit(win, lose, draw)

            if (next_win + 0.1 * draw) / (
                next_win + last_win + 0.2 * draw
            ) > update_threshold:
                log.info("Accept new model.")
                self.next_net.save_checkpoint(
                    folder=checkpoint_folder, filename=f"checkpoint_{i}.pth.tar"
                )
                self.next_net.save_checkpoint(
                    folder=checkpoint_folder, filename="best.pth.tar"
                )
                pit()
            else:
                log.info("Reject new model.")
                self.next_net.load_checkpoint(
                    folder=checkpoint_folder, filename="temp.pth.tar"
                )
                if len(pit_results) == 0:
                    pit()
                else:
                    _, win, lose, draw = pit_results[-1]
                    save_pit(win, lose, draw)
