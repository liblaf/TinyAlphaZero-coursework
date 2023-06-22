import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import coloredlogs

from . import GoNNetWrapper as nn
from .GoGame import GoGame as Game
from .Player import RandomPlayer
from .train_alphazero import Trainer
from .utils import set_seed_everywhere

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument(
    "--multiprocessing", action="store_true", default=False, dest="multiprocessing"
)

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")

BOARD_SIZE: int = 9

args: Dict[str, Any] = {
    "max_training_iter": 5000,  # 训练主循环最大迭代次数
    "selfplay_each_iter": 100,  # 每次训练迭代自我对弈次数
    "max_train_data_packs_len": 20,  # 最多保存最近的多少次训练迭代采集的数据
    "update_threshold": 0.00,  # 更新模型胜率阈值
    "update_match_cnt": 30,  # 计算更新模型胜率阈值的对弈次数
    "eval_match_cnt": 10,  # 每次更新模型后，进行评估的对弈次数
    "num_sims": 50,  # MCTS搜索的模拟次数
    "cpuct": 1,  # MCTS探索系数
    "cuda": True,  # 启用CUDA
    "checkpoint_folder": "./output",  # 模型保存路径
    "load_model": False,  # 是否加载模型
    "load_folder_file": ("output", "best.pth.tar"),  # 加载模型的路径
    "pit_with": RandomPlayer(game=Game(n=BOARD_SIZE), player=1),  # 评估时的对手
}


def main() -> None:
    cli_args: argparse.Namespace = parser.parse_args()
    args["multiprocessing"] = cli_args.multiprocessing

    load_model: bool = args["load_model"]
    load_folder_file: Tuple[Union[str, Path], str] = args["load_folder_file"]

    set_seed_everywhere(1)
    log.info("Loading %s...", Game.__name__)
    g = Game(BOARD_SIZE)

    log.info("Loading %s...", nn.__name__)
    nnet = nn(g)

    if load_model:
        log.info(
            'Loading checkpoint "%s/%s"...', load_folder_file[0], load_folder_file[1]
        )
        nnet.load_checkpoint(load_folder_file[0], load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Build Trainer...")
    c = Trainer(g, nnet, args)

    log.info("Start Training ...")
    c.train()


if __name__ == "__main__":
    main()
