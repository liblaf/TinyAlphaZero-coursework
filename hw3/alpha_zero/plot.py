from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from . import UPDATE_THRESHOLD

plt.switch_backend("agg")


def plot_win_rate(
    start_time: datetime,
    pit_results: List[Tuple[float, int, int, int]],
    output: Union[str, Path] = Path("output") / "win-rate.png",
) -> None:
    time_list: List[float] = []
    win_rate_list: List[float] = []
    undefeated_rate_list: List[float] = []
    for time, win, lose, draw in pit_results:
        delta: timedelta = datetime.fromtimestamp(time) - start_time
        time_list.append(delta.total_seconds())
        win_rate_list.append(win / (win + lose + draw))
        undefeated_rate_list.append((win + draw) / (win + lose + draw))
    plt.figure(dpi=300)
    plt.plot(np.array(time_list) / 3600.0, np.array(win_rate_list) * 100.0, label="Win")
    plt.plot(
        np.array(time_list) / 3600.0,
        np.array(undefeated_rate_list) * 100.0,
        label="Undefeated",
    )
    plt.xlabel("Time (hour)")
    plt.ylabel("Rate (%)")
    plt.legend(loc="best")
    plt.title("Win Rate Against Random")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_model_update_frequency(
    start_time: datetime,
    self_pit_results: List[Tuple[float, int, int, int]],
    output: Union[str, Path] = Path("output") / "model-update-frequency.png",
) -> None:
    time_list: List[float] = []
    update_frequency: List[float] = []
    last_update_time: datetime = start_time
    for time, win, lose, draw in self_pit_results:
        if (win + 0.1 * draw) / (win + lose + 0.2 * draw) <= UPDATE_THRESHOLD:
            continue
        current_time: datetime = datetime.fromtimestamp(time)
        delta: timedelta = current_time - start_time
        time_list.append(delta.total_seconds())
        update_frequency.append(1.0 / (current_time - last_update_time).total_seconds())
        last_update_time = current_time
    plt.figure(dpi=300)
    plt.plot(np.array(time_list) / 3600.0, np.array(update_frequency) * 3600.0)
    plt.xlabel("Time (hour)")
    plt.ylabel("Update Frequency (per hour)")
    plt.title("Update Frequency")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_loss(
    start_time: datetime,
    loss_history: List[Tuple[float, float]],
    output: Union[str, Path] = Path("output") / "loss.png",
) -> None:
    time_list: List[float] = []
    loss_list: List[float] = []
    for time, loss in loss_history:
        delta: timedelta = datetime.fromtimestamp(time) - start_time
        time_list.append(delta.total_seconds())
        loss_list.append(loss)
    plt.figure(dpi=300)
    plt.plot(np.array(time_list) / 3600.0, loss_list)
    plt.xlabel("Time (hour)")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
