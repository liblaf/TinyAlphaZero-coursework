import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from ..plot import plot_loss, plot_model_update_frequency, plot_win_rate


def test_plot_win_rate() -> None:
    start_time: datetime = datetime.now()
    pit_results: List[Tuple[float, int, int, int]] = []
    for i in range(100):
        pit_results.append(
            (
                datetime.now().timestamp(),
                random.randint(a=0, b=100),
                random.randint(a=0, b=100),
                random.randint(a=0, b=100),
            )
        )
    output_dir: Path = Path(tempfile.mkdtemp())
    try:
        output: Path = output_dir / "win-rate.png"
        plot_win_rate(start_time=start_time, pit_results=pit_results, output=output)
        assert output.exists()
    finally:
        shutil.rmtree(output_dir)


def test_plot_loss() -> None:
    start_time: datetime = datetime.now()
    loss_history: List[Tuple[float, float]] = []
    for i in range(100):
        loss_history.append((datetime.now().timestamp(), random.random()))
    output_dir: Path = Path(tempfile.mkdtemp())
    try:
        output: Path = output_dir / "loss.png"
        plot_loss(start_time=start_time, loss_history=loss_history, output=output)
        assert output.exists()
    finally:
        shutil.rmtree(output_dir)
