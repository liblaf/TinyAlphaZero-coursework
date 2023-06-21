import argparse
import logging

from anylearn.applications.quickstart import quick_train
from anylearn.config import init_sdk

logging.basicConfig(level=logging.INFO)
from . import NAME

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-u", "--username", dest="username")
parser.add_argument("-p", "--password", dest="password")
parser.add_argument("--disable-git", action="store_true", dest="disable_git")
parser.add_argument("--multiprocessing", action="store_true", dest="multiprocessing")


if __name__ == "__main__":
    args = parser.parse_args()
    init_sdk(
        cluster_address="https://anylearn.nelbds.cn",
        username=args.username,
        password=args.password,
        disable_git=args.disable_git,
    )
    train_task, _, _, _ = quick_train(
        algorithm_cloud_name=f"AlphaZero-{NAME}",
        algorithm_entrypoint=f'make train PYTHON_FLAGS="-OO" MULTIPROCESSING={args.multiprocessing}',  # 训练启动命令，可以是python、shell等等
        algorithm_force_update=True,
        algorithm_local_dir="./",
        algorithm_output="./output",
        hyperparams={},
        mirror_name="QUICKSTART_PYTORCH1.12.1_CUDA11",
        quota_group_request={
            "CPU": 8,
            "Memory": 32,
            "name": "SE2023",  # type: ignore
            "RTX-3090-shared": 1,
        },
        task_name=NAME
        + "-"
        + ("Multiprocessing" if args.multiprocessing else "Sequential"),
    )
    print(train_task)
