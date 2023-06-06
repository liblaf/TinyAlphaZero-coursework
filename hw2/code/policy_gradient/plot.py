import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", nargs="*", type=str, default=None, dest="input")
parser.add_argument("-s", "--seed", nargs="*", type=int, default=None, dest="seed")
parser.add_argument(
    "-p", "--prefix", type=Path, default=Path.cwd() / "assets", dest="prefix"
)


def main() -> None:
    args: argparse.Namespace = parser.parse_args()
    assert len(args.input) == len(args.seed)
    for i in range(len(args.input)):
        data: np.ndarray = np.loadtxt(args.input[i])
        assert data.shape[0] == 3
        episode, loss, reward = data
        assert isinstance(episode, np.ndarray)
        assert isinstance(loss, np.ndarray)
        assert isinstance(reward, np.ndarray)
        plt.figure(num=0, dpi=300)
        plt.plot(episode, loss, label=f"seed = {args.seed[i]}")
        plt.figure(num=1, dpi=300)
        plt.plot(episode, reward, label=f"seed = {args.seed[i]}")

    plt.figure(num=0, dpi=300)
    plt.legend(loc="best")
    plt.title("Reinforce Loss")
    plt.xlabel("episode")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(args.prefix / "loss.png")
    plt.figure(num=1, dpi=300)
    plt.legend(loc="best")
    plt.title("Reinforce Reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.tight_layout()
    plt.savefig(args.prefix / "reward.png")


if __name__ == "__main__":
    main()
