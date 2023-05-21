import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from eval import PLAYER_DICT, eval_parser, game, mcts_player, test_multi_match

parser: argparse.ArgumentParser = argparse.ArgumentParser(parents=[eval_parser])
parser.add_argument("-C", nargs="*", type=float, dest="C")
parser.add_argument("--train-iter", type=int, dest="train_iter")
parser.add_argument(
    "--param-prefix", default=Path.cwd() / "params", type=Path, dest="param_prefix"
)
parser.add_argument("--output", "-o", default=None, type=Path, dest="output")


def main(
    player1: str,
    player2: str,
    C_list: list[float],
    max_train_iter: int,
    param_prefix: Path = Path.cwd() / "params",
    output: Optional[Path] = None,
) -> None:
    mcts_player.eval()
    plt.figure(dpi=300)
    for C in C_list:
        x: np.ndarray = np.arange(1, max_train_iter + 1)
        y: np.ndarray = np.zeros_like(x, dtype=float)
        for i, train_iter in enumerate(x):
            mcts_player.game = game
            mcts_player.policy.C = C
            mcts_player.load_params(
                str(Path(param_prefix) / f"mcts_param_3x3_{C:.1f}_{train_iter}.pkl")
            )
            player1_win, player2_win, draw = test_multi_match(
                player1=PLAYER_DICT[player1],
                player2=PLAYER_DICT[player2],
                n_test=100,
                bilateral=False,
            )
            y[i] = ((player1_win if player1 == "MCTS" else player2_win) + draw) / (
                player1_win + player2_win + draw
            )
        plt.plot(x, y, label=f"C = {C:.1f}")
    plt.xlabel("Train Iter")
    plt.ylabel("MCTS Not Losing Rate")
    plt.legend(loc="best")
    plt.title(f"{player1} (offensive) vs. {player2} (defensive)")
    plt.tight_layout()
    if output:
        plt.savefig(output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        player1=args.player1,
        player2=args.player2,
        C_list=args.C,
        max_train_iter=args.train_iter,
        param_prefix=Path(args.param_prefix),
        output=Path(args.output),
    )
