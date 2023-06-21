from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt


def plot_win_rate(
    start_time: datetime,
    pit_results: List[Tuple[float, int, int, int]],
    output: Union[str, Path] = Path("output") / "win-rate.png",
) -> None:
    time_list: List[timedelta] = []
    win_rate_list: List[float] = []
    for time, win, lose, draw in pit_results:
        time_list.append(datetime.fromtimestamp(time) - start_time)
        win_rate_list.append(win / (win + lose + draw))
    plt.figure(dpi=300)
    plt.plot(time_list, win_rate_list)
    plt.xlabel("Time")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Against Random")
    plt.tight_layout()
    plt.savefig(output)


def plot_model_update_frequency(
    start_time: datetime,
    pit_results: List[Tuple[float, int, int, int]],
    output: Union[str, Path] = Path("output") / "model-update-frequency.png",
) -> None:
    pass


def plot_loss(
    start_time: datetime,
    loss_history: List[Tuple[float, float]],
    output: Union[str, Path] = Path("output") / "loss.png",
) -> None:
    time_list: List[timedelta] = []
    loss_list: List[float] = []
    for time, loss in loss_history:
        time_list.append(datetime.fromtimestamp(time) - start_time)
        loss_list.append(loss)
    plt.figure(dpi=300)
    plt.plot(time_list, loss_list)
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(output)
