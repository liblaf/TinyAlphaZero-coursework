import argparse
from pathlib import Path

from eval import *
from Player import *

parser = argparse.ArgumentParser(parents=[game_parser])
parser.add_argument("-C", dest="C", default=1.0, type=float)
parser.add_argument("--train-iter", dest="train_iter", default=500, type=int)
parser.add_argument(
    "--param-prefix", dest="param_prefix", default=Path.cwd() / "params", type=Path
)


TRAIN_ITER: int = 500
N_EVAL: int = 100

if __name__ == "__main__":
    args: argparse.Namespace = parser.parse_args()
    game.h = args.height
    game.w = args.width
    mcts_player.policy.C = args.C
    TRAIN_ITER: int = args.train_iter

    for t in range(TRAIN_ITER):
        mcts_player.train()
        score = single_match(mcts_player, mcts_player, game)
        print(f"Train iter-{t}: score={score}")

        print(f"Test iter-{t} mcts-random:")
        mcts_player.eval()
        player1, player2 = mcts_player, random_player

        # mcts offensive
        score = np.zeros(3)
        for _ in range(N_EVAL):
            score += np.array(single_match(player1, player2, game))
        offensive_win_rate = score[0] / N_EVAL * 100
        offensive_nolose_rate = (score[0] + score[1]) / N_EVAL * 100

        # mcts defensive
        score = np.zeros(3)
        for _ in range(N_EVAL):
            score += np.array(single_match(player2, player1, game))
        defensive_win_rate = score[2] / N_EVAL * 100
        defensive_nolose_rate = (score[2] + score[1]) / N_EVAL * 100

        print(
            f"  wining rate of mcts:     offensive={offensive_win_rate}, defensive={defensive_win_rate}"
        )
        print(
            f"  not losing rate of mcts: offensive={offensive_nolose_rate}, defensive={defensive_nolose_rate}"
        )

        mcts_player.save_params(
            args.param_prefix
            / f"mcts_param_{args.width}x{args.height}_{args.C}_{t + 1}.pkl"
        )
